"""模型集成 — 加权平均 + Hill Climbing前向选择

教学要点:
1. 集成的核心是模型多样性，而非模型数量
2. Hill Climbing是Kaggle中最简单有效的权重搜索方法
3. OOF (Out-of-Fold) 预测用于评估集成效果，避免过拟合
"""
import numpy as np
import pandas as pd
from src.utils.metrics import rmsle


def weighted_average(predictions_list: list, weights: list = None) -> np.ndarray:
    """加权平均集成。

    Args:
        predictions_list: 多个模型的预测数组列表
        weights: 权重列表，None则等权
    Returns:
        加权平均预测
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    weights = np.array(weights)
    weights = weights / weights.sum()  # 归一化

    result = np.zeros_like(predictions_list[0])
    for pred, w in zip(predictions_list, weights):
        result += w * pred
    return result


def hill_climbing(
    oof_preds_list: list,
    y_true: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """Hill Climbing前向选择最优集成权重。

    原理: 从空集成开始，每次贪心地添加一个模型的微小权重，
    如果RMSLE降低则保留，否则回退。最终得到每个模型的最优权重。

    Args:
        oof_preds_list: 多个模型的OOF预测列表
        y_true: 真实标签
        n_iterations: 迭代次数
        seed: 随机种子
    Returns:
        {"weights": array, "score": float, "blend_preds": array}
    """
    rng = np.random.RandomState(seed)
    n_models = len(oof_preds_list)
    n_samples = len(y_true)

    # 归一化OOF预测到非负
    oof_preds_list = [np.clip(p, 0, None) for p in oof_preds_list]

    best_weights = np.zeros(n_models)
    current_blend = np.zeros(n_samples)
    best_score = rmsle(y_true, np.clip(current_blend, 0, None))

    step_size = 0.01
    scores_history = [best_score]

    for i in range(n_iterations):
        # 随机选一个模型
        model_idx = rng.randint(0, n_models)
        # 随机选步长方向
        direction = rng.choice([-1, 1])
        delta = direction * step_size

        new_weights = best_weights.copy()
        new_weights[model_idx] += delta
        # 确保权重非负且总和为1
        new_weights = np.clip(new_weights, 0, None)
        weight_sum = new_weights.sum()
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        else:
            continue

        # 计算新集成预测
        new_blend = np.zeros(n_samples)
        for j, preds in enumerate(oof_preds_list):
            new_blend += new_weights[j] * preds

        new_score = rmsle(y_true, np.clip(new_blend, 0, None))

        if new_score < best_score:
            best_score = new_score
            best_weights = new_weights
            current_blend = new_blend
            scores_history.append(new_score)

    print(f"  Hill Climbing 最佳 RMSLE: {best_score:.5f}")
    print(f"  权重: {best_weights}")
    print(f"  有效迭代: {len(scores_history)-1}/{n_iterations}")

    return {
        "weights": best_weights,
        "score": best_score,
        "blend_preds": current_blend,
        "history": scores_history,
    }


def find_optimal_weights_grid(
    oof_preds_list: list,
    y_true: np.ndarray,
    n_steps: int = 20,
) -> dict:
    """网格搜索两模型最优权重 (适用于2-3个模型)。

    Args:
        oof_preds_list: OOF预测列表 (2-3个模型)
        y_true: 真实标签
        n_steps: 每个维度的搜索步数
    Returns:
        {"weights": array, "score": float, "blend_preds": array}
    """
    n_models = len(oof_preds_list)
    oof_preds_list = [np.clip(p, 0, None) for p in oof_preds_list]

    best_score = float("inf")
    best_weights = np.ones(n_models) / n_models
    best_blend = None

    if n_models == 2:
        for w0 in np.linspace(0, 1, n_steps + 1):
            w1 = 1 - w0
            blend = w0 * oof_preds_list[0] + w1 * oof_preds_list[1]
            score = rmsle(y_true, np.clip(blend, 0, None))
            if score < best_score:
                best_score = score
                best_weights = np.array([w0, w1])
                best_blend = blend

    elif n_models == 3:
        for w0 in np.linspace(0, 1, n_steps + 1):
            for w1 in np.linspace(0, 1 - w0, n_steps + 1):
                w2 = 1 - w0 - w1
                blend = (w0 * oof_preds_list[0]
                         + w1 * oof_preds_list[1]
                         + w2 * oof_preds_list[2])
                score = rmsle(y_true, np.clip(blend, 0, None))
                if score < best_score:
                    best_score = score
                    best_weights = np.array([w0, w1, w2])
                    best_blend = blend
    else:
        # 多于3个模型时用等权
        print(f"  警告: {n_models}个模型，网格搜索太慢，使用等权平均")
        best_weights = np.ones(n_models) / n_models
        best_blend = weighted_average(oof_preds_list)
        best_score = rmsle(y_true, np.clip(best_blend, 0, None))

    print(f"  网格搜索最佳 RMSLE: {best_score:.5f}")
    print(f"  权重: {best_weights}")

    return {
        "weights": best_weights,
        "score": best_score,
        "blend_preds": best_blend,
    }
