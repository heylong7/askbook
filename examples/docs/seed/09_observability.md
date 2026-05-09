# 可观测性与链路追踪

askbook 通过异步 JSONL 链路追踪实现全链路可观测，无需外部 APM 服务。每个 Pipeline 节点自动上报 span 事件，包含耗时、Token 消耗和成本数据。

## JSONL Trace 格式

Trace 事件追加写入按日分片的 JSONL 文件：`~/.askbook/traces/YYYY-MM-DD.jsonl`。每个事件是自包含的 JSON 对象：

- **trace_id**：标识单次管线执行的 UUID
- **span_id**：本次节点执行的唯一标识
- **parent_span_id**：父节点（根节点为 null）
- **event_type**：`span_start`、`span_end` 或 `error`
- **node_name**：Pipeline 节点标识
- **timestamp_utc**：ISO 8601 时间戳
- **duration_ms**：耗时（仅 span_end）
- **tags**：节点元数据（Token 数、chunk 数、模型信息等）
- **error**：异常消息（仅 error 事件）

## 异步 Trace Writer

AsyncTraceWriter 使用后台守护线程 + 内部队列。span 事件在管线执行期间入队，异步写入磁盘，不阻塞热路径。`atexit` 处理器确保进程退出前排空队列。

配置项：
- `observability.enabled`（默认 true）：设为 false 可完全跳过 trace 写入
- `observability.trace_dir`：JSONL 文件存储目录
- `observability.retention_days`（默认 7）：自动删除 N 天前的 trace 文件
- `observability.pii_redaction`（默认 true）：写入前脱敏手机号、邮箱和 Token
- `observability.flush_interval_seconds`（默认 1.0）：队列刷新间隔

## PII 脱敏

Trace 事件写入磁盘前，经过正则脱敏过滤器屏蔽：
- 中国大陆手机号（11 位）
- 邮箱地址
- API Token 模式

避免敏感用户数据泄漏到 trace 日志中。

## Pipeline 节点的自动 Trace

BasePipelineNode 的 `__call__` 方法自动创建 span 上下文管理器。子类只需实现 `run()`，基类处理 span 生命周期（start、end、error）。可选的 `before_run()` 和 `after_run()` 钩子让节点向 span 附加自定义标签。

QuerySpan 中的 `original_query` 字段保留用户改写前的原始输入，Dashboard 的 Rewrite Diff 视图利用该字段检测改写漂移。
