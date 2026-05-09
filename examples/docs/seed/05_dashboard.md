# Streamlit 看板

askbook 内置 Streamlit 监控看板，提供系统监控、数据浏览和链路追踪功能。无需外部数据库或 APM 服务，直接读取 ChromaDB 元数据和 JSONL trace 文件。

## 启动看板

```bash
uv run askbook dashboard           # 默认端口 8501
uv run askbook dashboard --port 9000
```

打开 `http://localhost:8501` 访问五页面看板。

## 页面 1 —— 系统总览

实时展示系统健康状态和核心指标：
- 健康检查卡片（LLM / ChromaDB / BM25 三组件状态）
- 今日查询量、P50 延迟、Token 消耗总量
- Collection 级别 chunk 数量统计

## 页面 2 —— 数据浏览

ChromaDB collection 的只读浏览器：
- 选择 collection 查看文档列表（doc_id、源文件路径、chunk 数、入库时间）
- 点击文档展开查看各 chunk 内容预览

## 页面 3 —— Pipeline 监控

两个标签页：
- **Ingestion 时间线**：解析 JSONL trace，展示每次入库的文档数、chunk 统计和耗时。Plotly 甘特图按节点展示单次 Pipeline 执行的时间分布
- **Query Rewrite 审计**：对比原始查询与改写后查询，检测改写漂移

## 页面 4 —— Trace 查看器

解析最近 7 天的 JSONL trace 文件：
- 筛选控件（天数、事件类型、节点名）
- 事件表格（trace_id、span_id、节点名、耗时、Token 消耗）
- 点击展开 JSON 详情和水瀑布视图

## 页面 5 —— 评估结果

- Harness 四项健康指标（完成率、重试率、一次通过率、单任务成本）
- 检索指标趋势图（Recall@5、MRR、NDCG）
- 用户反馈收集入口（赞/踩按钮）

## 缓存策略

使用 `@st.cache_data`，以 JSONL 文件 mtime 作为缓存键。新 trace 写入后 mtime 变化，缓存自动失效，实现近实时更新。
