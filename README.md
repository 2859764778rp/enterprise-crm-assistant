# 企业智能客服助手

面向企业内部销售/客服场景的智能问答系统：通过「意图路由 + 知识库 RAG + Function Calling 工具调用」实现资料自助检索、订单/客户记录自然语言查询与个性化引导，减少重复性人工沟通。

## 核心功能

- **意图识别 + 条件路由**：业务咨询 → 知识库 RAG；其他问题 → Agent 工具调用（LLM Function Calling）
- **轻量级本地向量检索模块**：基于 numpy 余弦相似度的本地向量库（线程安全、磁盘持久化），支撑双知识库在线增删改查
- **Function Calling 工具循环**（最多 4 轮、无工具调用即终止、工具异常降级）：
  - `web_search`：联网检索
  - `NL2SQL`：自然语言查库（只读 SELECT 白名单 + 关键字/表名拦截 + 自动 LIMIT）
- **用户画像**：基于意图历史与标签（本地 SQLite 存储）生成个性化引导
- **流式 NDJSON 接口**：支持流式输出；基于 Little 定律估算并发承载（QPS × 响应时间）

## 目录结构

```
AI CRM/
├── app.py                # Flask 入口（流式 NDJSON 接口）
├── chat.py               # 对话编排
├── crm_core.py           # 业务查询核心
├── intent.py             # 意图识别与路由
├── tools.py              # Function Calling 工具（web_search / NL2SQL）
├── vectorstore.py        # 轻量级本地向量检索模块
├── embeddings.py         # 向量化（TF-IDF / bge）
├── database.py           # SQLite（WAL）会话与画像存储
├── models.py             # 数据模型
├── seed_data.py          # 知识库初始化
├── gradio_interface.py   # Gradio 演示界面
├── API.md                # 接口说明
└── requirements.txt      # 依赖清单
```

## 运行方式

```bash
# Windows PowerShell
cd "AI CRM"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 配置 .env（DEEPSEEK_API_KEY 等）
# 初始化知识库并启动
python seed_data.py
python app.py
```

## 说明

- 本项目为个人学习/实习实践项目，代码结构可自由参考。
- NL2SQL 采用只读白名单 + 关键字/表名拦截 + 自动 LIMIT 的安全策略；更多安全边界说明见 `AI CRM/API.md`。
