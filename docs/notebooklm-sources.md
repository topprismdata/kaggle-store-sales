# NotebookLM 竞赛智囊 — 资料收集方法记录

> 本文档记录了获取 Kaggle 竞赛资料并上传到 NotebookLM 的完整流程。
> 这是 agentic 竞赛流程中 Phase 0 的核心教学内容。

## 一、获取 Kaggle Notebook 的方法

### 方法 1: Kaggle CLI (首选)

**优点**: 无需登录、数据结构化、100%可靠、可排序分页

```bash
# 获取 TOP notebooks 列表（按投票数排序）
kaggle kernels list --competition store-sales-time-series-forecasting --sort-by voteCount --page-size 20

# 下载单个 notebook
kaggle kernels pull <author>/<slug> -p <output_dir>

# 批量下载示例
kaggle kernels pull ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide -p data/notebooks/ekrembayar/
```

**关键注意事项**:
- 每个 kernel 需要单独的下载目录
- 下载的是 `.ipynb` 格式，NotebookLM 不支持直接上传

### 方法 2: ipynb → Markdown 转换

NotebookLM 只支持 `.md`, `.txt`, `.pdf`, `.docx` 等格式。
直接上传 `.ipynb` 会返回 `400 Bad Request`。

```python
import json
from pathlib import Path

def convert_ipynb_to_md(ipynb_path):
    """Jupyter notebook → Markdown"""
    with open(ipynb_path, 'r') as f:
        nb = json.load(f)

    md_lines = [f"# {ipynb_path.stem}\n"]

    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type', 'code')
        source = ''.join(cell.get('source', []))

        if cell_type == 'markdown':
            md_lines.append(source)
        elif cell_type == 'code':
            md_lines.append(f"\n```python\n{source}\n```\n")

    return '\n'.join(md_lines)

# 批量转换
for ipynb in Path("/tmp/store_sales_notebooks").rglob("*.ipynb"):
    md_content = convert_ipynb_to_md(ipynb)
    author = ipynb.parent.name
    output = Path("/tmp/store_sales_md/") / f"{author}_{ipynb.stem}.md"
    output.write_text(md_content, encoding='utf-8')
```

### 方法 3: 批量上传到 NotebookLM

```bash
# 创建 notebook
notebooklm create "Store Sales Time Series Forecasting 竞赛资料"

# 设置当前 notebook 上下文
notebooklm use <NOTEBOOK_ID>

# 批量上传（注意 3 秒间隔避免 rate limiting）
for md_file in /tmp/store_sales_md/*.md; do
    filename=$(basename "$md_file")
    title="${filename%.md}"
    notebooklm source add "$md_file" --title "$title"
    sleep 3
done
```

## 二、获取 Kaggle 论坛讨论的方法

### 方法 1: Web Search (首选)

**优点**: 无需登录、速度快、覆盖面广

```
搜索: "store sales time series forecasting kaggle 1st place solution"
搜索: "store sales kaggle best RMSLE technique"
```

### 方法 2: GitHub 镜像仓库

Kaggle 竞赛页面是 JS 渲染的，但 GitHub 上有镜像仓库包含完整的竞赛说明。
例如: https://github.com/Tech-i-s/techis-ds-kaggle-store_sales_time_series_forcasting

### 方法 3: MCP Web Reader

通过 MCP 工具获取网页内容。对 GitHub 和 Medium 页面有效，Kaggle JS 渲染页面可能失败（返回 reCAPTCHA 或空内容）。

## 三、其他资料来源

| 资料类型 | 获取方法 | 优先级 |
|---------|---------|--------|
| 竞赛规则和数据说明 | GitHub README (Tech-i-s repo) | 高 |
| TOP Notebooks | `kaggle kernels list` + `kaggle kernels pull` | 高 |
| 获奖方案解析 | Web Search + Web Reader | 高 |
| 论坛热门讨论 | Web Search 或 Playwright | 中 |
| Medium/YouTube 教程 | Web Search + Web Reader | 中 |

## 四、已上传资料清单

### NotebookLM Notebook 信息

- **Notebook ID**: `9d974d8f-551a-466b-b07a-1b550b978141`
- **Title**: Store Sales Time Series Forecasting 竞赛资料
- **Created**: 2026-04-06
- **总资料数**: 12 个 source

### 资料清单

| # | 文件名 | 来源 | 描述 | 大小 |
|---|--------|------|------|------|
| 1 | competition_overview.md | 竞赛官方页面 (via GitHub) | 竞赛目标、评估指标 RMSLE、提交格式、Top Approaches 总结 | 5.1KB |
| 2 | competition_data_description.md | 竞赛官方页面 (via GitHub) | 数据文件说明、字段描述、特征工程建议 | 4.9KB |
| 3 | winning_solutions.md | Web Search + 多源整理 | 获奖方案汇总、最佳技术、RMSLE 优化技巧 | 6.1KB |
| 4 | ekrembayar_store-sales-ts-forecasting-a-comprehensive-guide.md | Kaggle Notebook | 全面的时间序列预测指南，EDA + 特征工程 + 建模 | 39KB |
| 5 | kashishrastogi_store-sales-analysis-time-serie.md | Kaggle Notebook | Store Sales 分析和时间序列方法 | 19KB |
| 6 | ferdinandberr_darts-forecasting-deep-learning-global-models.md | Kaggle Notebook | DARTS 库深度学习模型 (N-BEATS, TFT, N-HiTS) | 98KB |
| 7 | ivanlydkin_time-series-course-a-practical-guide.md | Kaggle Notebook | 时间序列课程实战指南 | 67KB |
| 8 | odins0n_exploring-time-series-plots-beginners-guide.md | Kaggle Notebook | 时间序列可视化入门指南 | 7.3KB |
| 9 | hardikgarg03_store-sales.md | Kaggle Notebook | Store Sales 基础方案 | 5.2KB |
| 10 | maricinnamon_store-sales-time-series-forecast-visualization.md | Kaggle Notebook | Store Sales 可视化和预测分析 | 34KB |
| 11 | chongzhenjie_ecuador-store-sales-global-forecasting-lightgbm.md | Kaggle Notebook | 厄瓜多尔商店销售全局 LightGBM 预测 | 78KB |
| 12 | howoojang_first-kaggle-notebook-following-ts-tutorial.md | Kaggle Notebook | 跟随时间序列教程的第一个 Kaggle Notebook | 32KB |

## 五、RAG 质量验证

上传完成后，使用以下查询测试 NotebookLM 智囊的 RAG 质量：

### 查询 1: "这个竞赛的评估指标是什么？"
**结果**: PASS
- 正确回答了 RMSLE (Root Mean Squared Logarithmic Error)
- 补充说明了为什么使用 RMSLE (惩罚低估、与绝对规模无关)
- 引用了多个来源 [1]

### 查询 2: "Store Sales 竞赛最好的模型是什么？"
**结果**: PASS
- 详细说明了 LightGBM/XGBoost 是最佳模型
- 介绍了混合模型策略 (线性模型 + GBDT)
- 提到了深度学习方法 (DARTS) 表现不及树模型
- 强调了特征工程比模型选择更重要
- 引用了多个来源 [1-9]

### 查询 3: "油价数据怎么处理？"
**结果**: PASS
- 说明了周末/节假日缺失值的处理 (forward-fill, 线性插值)
- 介绍了滚动统计特征 (7/14/28天滚动均值)
- 提到了70美元阈值的有趣发现
- 给出了完整的管道流程建议
- 引用了多个来源 [1-13]

### 查询 4: "lag特征怎么构建？"
**结果**: PASS
- 说明了分组维度 (store_nbr, family)
- 列出了常用滞后步长 (1, 7, 14, 28, 365 天)
- 提到了使用 PACF 图辅助选择
- 详细解释了数据泄露的注意事项
- 说明了16天预测窗口的匹配策略
- 引用了多个来源 [1-17]

### 验证总结

所有 4 个查询都返回了高质量、多源引用的详细答案，RAG 质量验证通过。

## 六、竞赛过程中如何使用智囊

1. **特征工程阶段**: 询问 "有哪些有效的特征工程方法？" 获取灵感和代码片段
2. **模型选择阶段**: 询问 "LightGBM 和 XGBoost 哪个更适合这个竞赛？" 获取对比分析
3. **调参阶段**: 询问 "LightGBM 的超参数怎么调？" 获取经验参数
4. **问题排查**: 询问 "为什么我的 RMSLE 很高？" 获取常见错误和改进建议
5. **特定技术**: 询问 "如何处理零销量？" 或 "如何构建节假日特征？" 获取具体方案
6. **代码参考**: 询问 "show me the lag feature code" 获取代码实现

### 使用方式

```bash
# 切换到竞赛智囊 notebook
notebooklm use 9d974d8f

# 提问
notebooklm ask "你的问题"

# 查看对话历史
notebooklm history
```
