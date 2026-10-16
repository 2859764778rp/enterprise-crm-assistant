# AI CRM 接口文档与技术报告

本文对应作业报告的四项要求：
1. 完整代码 + 运行说明（见 `README.md` 与各源文件）
2. function calling 工具说明
3. RAG 存储内容与优化调研
4. 接口文档 + 服务并发预估

---

## 一、接口文档

服务基址：`http://<host>:5000`
数据格式：请求/响应均为 JSON（`/chat` 响应为流式 NDJSON）。

### 1. POST /chat — 对话（流式）

请求体：
```json
{ "phone_number": "13911112222", "query": "AI Agent 训练营多少钱" }
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| phone_number | string | 是 | 用户唯一标识（对应伪代码入参） |
| query | string | 是 | 用户本轮输入 |

响应：`Content-Type: application/x-ndjson`，逐行返回：
```
{"delta": "您好，"}
{"delta": "AI Agent..."}
{"done": true}
```
出错时该行为 `{"error": "..."}`。

错误码：`400` 缺少 phone_number 或 query。

### 2. POST /vectorstore/update — 向量库更新（对外开放）

请求体：
```json
{
  "store": "course",
  "documents": ["新增课程：AI Agent 高级开发训练营，价格 9800 元。"],
  "metadatas": [{"source": "manual"}]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| store | string | 是 | `market` 或 `course` |
| documents | string[] | 是 | 待入库文本，非空 |
| metadatas | object[] | 否 | 与 documents 等长的元数据 |

响应：
```json
{ "store": "course", "added": 1, "ids": ["..."], "total": 7 }
```

### 3. GET /vectorstore/stats — 各库文档数
```json
{ "market": 5, "course": 7 }
```

### 4. GET /health — 健康检查
```json
{ "status": "ok" }
```

---

## 二、Function Calling 工具说明

### 2.1 搭建基本流程

1. **定义工具 schema**（`tools.py` 的 `TOOL_SCHEMAS`）：用 OpenAI function calling 规范
   描述每个工具的 `name`、`description`、`parameters`（JSON Schema）。
2. **实现工具函数**：`web_search(query)`、`query_user_db(question)`。
3. **注册执行器**：`execute_tool(name, args)` 按名字分发到对应函数。
4. **对话循环**（`models.py` 的 `chat_with_tools`）：
   - 把 `tools` 连同 messages 发给模型，`tool_choice="auto"`；
   - 若模型返回 `tool_calls`，逐个执行并以 `role="tool"` 回填结果；
   - 循环直到模型不再调用工具或达到 `max_rounds`，返回最终文本。

### 2.2 定义的函数功能

| 工具 | 功能 | 说明 |
|------|------|------|
| `web_search(query)` | 网络搜索插件 | 用 DuckDuckGo 免 key 接口检索实时信息；网络不可用时返回占位说明，不中断对话 |
| `query_user_db(question)` | SQL Agent | 自然语言 → 受限只读 SQL → 查询 SQLite 用户数据 → 文本结果 |

**SQL Agent 安全设计**：只允许 `SELECT`；正则拦截 `insert/update/delete/drop/alter/create` 等写/DDL 关键字；
表名白名单（`chat_history`/`user_profile`）；自动追加 `LIMIT` 限制返回行数。已实测 `DROP` 语句被拒绝。

---

## 三、RAG 存储内容与优化调研

### 3.1 本项目 RAG 存了什么

两个本地向量库（`data/vecstore/*.npy` + `*.json`）：
- **market**：AI 职业咨询知识（岗位方向、转行路径、薪资、技能要求等）
- **course**：课程咨询知识（大纲、周期、价格、报名方式、师资、结课项目等）

存储结构：每条文档保存 `{id, text, meta}`，向量矩阵与文档顺序对齐；
向量经 L2 归一化，检索用点积即余弦相似度，取 Top-K 注入 prompt。

### 3.2 RAG 常见优化方式及与本项目的融合

| 优化方向 | 说明 | 如何融合到本项目 |
|----------|------|------------------|
| **更强 embedding** | bge/m3e/gte 等中文模型 | 已在 `embeddings.py` 预留 sentence-transformers 入口，装上即用，无需改上层 |
| **混合检索(Hybrid)** | 向量 + BM25/关键词融合排序 | TF-IDF 逻辑已在，可与语义分数加权合并 |
| **Rerank 重排** | 用 cross-encoder 对召回结果精排 | 在 `vectorstore.query` 后接一层 rerank 再截 Top-K |
| **Chunking 优化** | 按语义/标题切分、重叠窗口 | 入库前对长文档切分后再 `add` |
| **Query 改写/扩展** | 用 LLM 改写用户问题提升召回 | 检索前先用 `chat_once` 生成扩展查询 |
| **元数据过滤** | 按标签/来源缩小检索范围 | `meta` 字段已保存，可加过滤条件 |
| **多路召回融合** | 多库/多策略召回后融合 | market/course 可并行召回再融合 |
| **引用与防幻觉** | 返回来源、无命中时兜底 | 已在 prompt 中约束「优先据知识库作答」，可进一步加引用标注 |

### 3.3 换更强的向量库

当前用 numpy 自建（Python 3.13 下最稳）。数据量增大后可平滑迁移到
FAISS（本地、快）或 Chroma/Milvus（功能全），只需替换 `LocalVectorStore` 实现，
`add/update/query` 接口保持不变。

---

## 四、服务并发量预估

### 4.1 瓶颈分析

单次 `/chat` 请求的耗时构成（实测量级）：
- 意图识别（1 次 DeepSeek 非流式调用）：约 0.4–1s
- RAG 检索（本地 numpy）：< 10ms，可忽略
- 生成回复（1 次 DeepSeek 流式调用）：约 1–4s（取决于回复长度）
- function calling 分支额外 1–3 次模型往返

**结论：瓶颈是 DeepSeek API 的网络往返与生成耗时，而非本地 CPU。**
本地组件（SQLite、numpy 向量检索）耗时极低，几乎不构成瓶颈。

### 4.2 并发估算（Little's 定律）

并发数 ≈ 吞吐(QPS) × 平均响应时间(RT)。设单请求平均 RT ≈ 3s：

| 部署方式 | 并发能力 | 说明 |
|----------|----------|------|
| Flask 开发服务器(threaded) | ~10–20 并发 | 仅演示，GIL + 单进程限制 |
| gunicorn 4×worker | ~50–80 并发 | 每 worker 多线程，I/O 密集受益明显 |
| gunicorn/uvicorn 多进程 + 异步 | 数百并发 | 受限于 DeepSeek API 的账号限流(RPM/TPM) |

**关键约束**：由于计算实际发生在 DeepSeek 侧，系统并发上限最终由
**DeepSeek 账号的速率限制（RPM/TPM）** 决定，而非本机资源。

### 4.3 扩容建议

- Web 层：换生产 WSGI（gunicorn），按 `2×CPU+1` 配置 worker，因 I/O 密集可加大线程数。
- 数据层：SQLite 已开 WAL 支持并发读；高并发写场景建议迁移到 PostgreSQL。
- 模型层：接入请求队列 + 限流，对超限做排队/降级；命中率高的问题加缓存。
- 向量层：数据量大后迁 FAISS/Milvus，检索仍非瓶颈。
- 部署多实例 + 负载均衡横向扩展；SQLite 换成共享数据库后可无状态水平扩容。

---

## 附：与原项目的复用关系

| 原项目资产 | 在 AI CRM 中的角色 |
|------------|--------------------|
| `models.py`（DeepSeek + 流式） | 保留并扩展出 `chat_once` / `chat_with_tools` |
| `.env`（DeepSeek Key） | 原样复用，全程仅此一个 Key |
| `chat.py` 的历史抽象 | 逻辑迁移进 `crm_core.py` + `database.py` 持久化 |
| Flask 依赖 | 用于 `app.py` 对外封装 |
| `gradio_interface.py` | 保留为本地调试 UI |


