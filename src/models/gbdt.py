"""GBDT模型训练 — LightGBM/XGBoost/CatBoost with time-series CV"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
from src.config import ModelConfig, XGBoostConfig, CatBoostConfig, CVConfig, MODELS
from src.utils.metrics import rmsle


def time_series_split(df: pd.DataFrame, cfg: CVConfig):
    """时间序列交叉验证分割器。
    从train_end往前，每次取val_days天作为验证集。
    Yields: (train_idx, val_idx)
    """
    dates = df["date"].sort_values().unique()
    n_splits = cfg.n_folds
    val_days = cfg.val_days

    for i in range(n_splits):
        val_end_idx = len(dates) - 1 - i * val_days
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - 1

        if train_end_idx < 0 or val_start_idx < 0:
            break

        val_dates = dates[val_start_idx : val_end_idx + 1]
        train_dates = dates[: train_end_idx + 1]

        train_idx = df[df["date"].isin(train_dates)].index
        val_idx = df[df["date"].isin(val_dates)].index

        yield train_idx, val_idx


def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: list,
    cfg: ModelConfig = None,
    cv_cfg: CVConfig = None,
    log_transform: bool = True,
) -> dict:
    """训练LightGBM模型，时间序列CV评估。
    log_transform=True时对sales做log1p变换（匹配RMSLE目标）。
    返回: {"model": model, "cv_scores": [...], "mean_cv": float, "oof_preds": array}
    """
    if cfg is None:
        cfg = ModelConfig()
    if cv_cfg is None:
        cv_cfg = CVConfig()

    df = df.copy().reset_index(drop=True)
    target = df["sales"].values.copy()

    if log_transform:
        target = np.log1p(target)

    cv_scores = []
    oof_preds = np.zeros(len(df))
    models = []

    for fold, (train_idx, val_idx) in enumerate(time_series_split(df, cv_cfg)):
        X_train = df.loc[train_idx, feature_cols]
        y_train = target[train_idx]
        X_val = df.loc[val_idx, feature_cols]
        y_val = target[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            {
                "objective": cfg.objective,
                "metric": cfg.metric,
                "boosting_type": cfg.boosting_type,
                "num_leaves": cfg.num_leaves,
                "min_child_samples": cfg.min_child_samples,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "reg_alpha": cfg.reg_alpha,
                "reg_lambda": cfg.reg_lambda,
                "learning_rate": cfg.learning_rate,
                "verbose": -1,
                "random_state": cfg.random_state,
            },
            train_data,
            num_boost_round=cfg.n_estimators,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        val_pred = model.predict(X_val)
        if log_transform:
            val_pred_orig = np.expm1(val_pred)
            y_val_orig = np.expm1(y_val)
        else:
            val_pred_orig = val_pred
            y_val_orig = y_val

        val_pred_orig = np.clip(val_pred_orig, 0, None)

        score = rmsle(y_val_orig, val_pred_orig)
        cv_scores.append(score)
        oof_preds[val_idx] = val_pred_orig
        models.append(model)

        print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration})")

    mean_cv = np.mean(cv_scores)
    print(f"\n  Mean CV RMSLE: {mean_cv:.5f} (+/- {np.std(cv_scores):.5f})")

    # 用全部数据重训最终模型
    X_all = df[feature_cols]
    y_all = target
    all_data = lgb.Dataset(X_all, label=y_all)

    final_model = lgb.train(
        {
            "objective": cfg.objective,
            "metric": cfg.metric,
            "boosting_type": cfg.boosting_type,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "learning_rate": cfg.learning_rate,
            "verbose": -1,
            "random_state": cfg.random_state,
        },
        all_data,
        num_boost_round=max(m.best_iteration for m in models),
    )

    return {
        "model": final_model,
        "cv_models": models,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "oof_preds": oof_preds,
    }


def predict(model, df: pd.DataFrame, feature_cols: list, log_transform: bool = True) -> np.ndarray:
    """用模型预测，如果log_transform则做expm1反变换"""
    preds = model.predict(df[feature_cols])
    if log_transform:
        preds = np.expm1(preds)
    return np.clip(preds, 0, None)


def predict_cv_models(cv_models: list, df: pd.DataFrame, feature_cols: list,
                       log_transform: bool = True) -> np.ndarray:
    """用CV models的平均预测 — 比final model更稳定，避免过拟合。
    教学要点: final model在全量数据上训练但无early stopping，容易过拟合。
    CV models有early stopping保护，预测更可靠。
    """
    preds_list = []
    for m in cv_models:
        p = m.predict(df[feature_cols])
        if log_transform:
            p = np.expm1(p)
        preds_list.append(np.clip(p, 0, None))
    return np.mean(preds_list, axis=0)


def generate_submission(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    output_path: str = None,
    log_transform: bool = True,
) -> pd.DataFrame:
    """生成提交文件"""
    from src.config import SUBMISSIONS

    preds = predict(model, test_df, feature_cols, log_transform=log_transform)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "sales": preds,
    })

    from pathlib import Path as _Path
    if output_path is None:
        output_path = SUBMISSIONS / "submission_baseline.csv"
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Sales range: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
    print(f"  Sales mean: {submission['sales'].mean():.2f}")

    return submission


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list,
    cfg: XGBoostConfig = None,
    cv_cfg: CVConfig = None,
    log_transform: bool = True,
) -> dict:
    """训练XGBoost模型，时间序列CV评估。
    返回: {"model": model, "cv_scores": [...], "mean_cv": float, "oof_preds": array}
    """
    if cfg is None:
        cfg = XGBoostConfig()
    if cv_cfg is None:
        cv_cfg = CVConfig()

    df = df.copy().reset_index(drop=True)
    target = df["sales"].values.copy()

    if log_transform:
        target = np.log1p(target)

    cv_scores = []
    oof_preds = np.zeros(len(df))
    models = []
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(time_series_split(df, cv_cfg)):
        X_train = df.loc[train_idx, feature_cols]
        y_train = target[train_idx]
        X_val = df.loc[val_idx, feature_cols]
        y_val = target[val_idx]

        model = xgb.XGBRegressor(
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            tree_method=cfg.tree_method,
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        if log_transform:
            val_pred_orig = np.expm1(val_pred)
            y_val_orig = np.expm1(y_val)
        else:
            val_pred_orig = val_pred
            y_val_orig = y_val

        val_pred_orig = np.clip(val_pred_orig, 0, None)

        score = rmsle(y_val_orig, val_pred_orig)
        cv_scores.append(score)
        oof_preds[val_idx] = val_pred_orig
        models.append(model)
        best_iters.append(model.best_iteration)

        print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration})")

    mean_cv = np.mean(cv_scores)
    print(f"\n  Mean CV RMSLE: {mean_cv:.5f} (+/- {np.std(cv_scores):.5f})")

    # 用全部数据重训最终模型
    X_all = df[feature_cols]
    y_all = target
    final_model = xgb.XGBRegressor(
        objective=cfg.objective,
        eval_metric=cfg.eval_metric,
        n_estimators=max(best_iters),
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        tree_method=cfg.tree_method,
        verbosity=0,
    )
    final_model.fit(X_all, y_all, verbose=False)

    return {
        "model": final_model,
        "cv_models": models,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "oof_preds": oof_preds,
    }


def train_catboost(
    df: pd.DataFrame,
    feature_cols: list,
    cfg: CatBoostConfig = None,
    cv_cfg: CVConfig = None,
    log_transform: bool = True,
) -> dict:
    """训练CatBoost模型，时间序列CV评估。
    返回: {"model": model, "cv_scores": [...], "mean_cv": float, "oof_preds": array}
    """
    if cfg is None:
        cfg = CatBoostConfig()
    if cv_cfg is None:
        cv_cfg = CVConfig()

    df = df.copy().reset_index(drop=True)
    target = df["sales"].values.copy()

    if log_transform:
        target = np.log1p(target)

    cv_scores = []
    oof_preds = np.zeros(len(df))
    models = []
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(time_series_split(df, cv_cfg)):
        X_train = df.loc[train_idx, feature_cols]
        y_train = target[train_idx]
        X_val = df.loc[val_idx, feature_cols]
        y_val = target[val_idx]

        model = CatBoostRegressor(
            loss_function=cfg.loss_function,
            iterations=cfg.iterations,
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            l2_leaf_reg=cfg.l2_leaf_reg,
            random_seed=cfg.random_seed,
            bootstrap_type=cfg.bootstrap_type,
            subsample=cfg.subsample,
            colsample_bylevel=cfg.colsample_bylevel,
            verbose=cfg.verbose,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        val_pred = model.predict(X_val)
        if log_transform:
            val_pred_orig = np.expm1(val_pred)
            y_val_orig = np.expm1(y_val)
        else:
            val_pred_orig = val_pred
            y_val_orig = y_val

        val_pred_orig = np.clip(val_pred_orig, 0, None)

        score = rmsle(y_val_orig, val_pred_orig)
        cv_scores.append(score)
        oof_preds[val_idx] = val_pred_orig
        models.append(model)
        best_iters.append(model.best_iteration_)

        print(f"  Fold {fold+1}: RMSLE = {score:.5f} (best_iter={model.best_iteration_})")

    mean_cv = np.mean(cv_scores)
    print(f"\n  Mean CV RMSLE: {mean_cv:.5f} (+/- {np.std(cv_scores):.5f})")

    # 用全部数据重训最终模型
    X_all = df[feature_cols]
    y_all = target
    final_model = CatBoostRegressor(
        loss_function=cfg.loss_function,
        iterations=max(best_iters),
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_seed=cfg.random_seed,
        bootstrap_type=cfg.bootstrap_type,
        subsample=cfg.subsample,
        colsample_bylevel=cfg.colsample_bylevel,
        verbose=False,
    )
    final_model.fit(X_all, y_all, verbose=False)

    return {
        "model": final_model,
        "cv_models": models,
        "cv_scores": cv_scores,
        "mean_cv": mean_cv,
        "oof_preds": oof_preds,
    }
