from __future__ import annotations

import json
import re
import sqlite3

import requests

from config import DB_PATH, CHAT_MODEL, logger
from database import connect_to_database

def web_search(query: str, max_results: int = 3) -> str:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        data = resp.json()
    except Exception as e:
        logger.warning("web_search 失败: %s", e)
        return f"[网络搜索暂不可用] 无法检索“{query}”。"

    parts = []
    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
    for topic in (data.get("RelatedTopics") or [])[:max_results]:
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(topic["Text"])
    if not parts:
        return f"未检索到关于“{query}”的明确结果。"
    return "\n".join(f"- {p}" for p in parts[:max_results])


_ALLOWED_TABLES = {"chat_history", "user_profile"}
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|pragma|;)\b", re.I
)

_SQL_SYSTEM = (
    "你是 SQL 生成助手。数据库是 SQLite，只有两张表：\n"
    "chat_history(id, phone_number, role, content, ts)\n"
    "user_profile(phone_number, name, intent_history, tags, notes, created_at, updated_at)\n"
    "请根据用户问题，只生成一条 SELECT 查询语句（禁止任何修改语句），"
    "不要解释，不要用代码块，直接输出 SQL。"
)


def _nl_to_sql(question: str) -> str:
    from models import chat_once 
    sql = chat_once(
        [{"role": "system", "content": _SQL_SYSTEM},
         {"role": "user", "content": question}],
        model=CHAT_MODEL, temperature=0.0,
    )
    sql = re.sub(r"^```[a-zA-Z]*|```$", "", sql.strip()).strip()
    return sql


def _is_safe_select(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select"):
        return False
    if _FORBIDDEN.search(s + " "):
        return False
    
    for tbl in re.findall(r"\bfrom\s+([a-zA-Z_]+)", s, re.I):
        if tbl.lower() not in _ALLOWED_TABLES:
            return False
    return True


def query_user_db(question: str) -> str:
    try:
        sql = _nl_to_sql(question)
    except Exception as e:
        return f"[SQL 生成失败] {e}"

    if not _is_safe_select(sql):
        logger.warning("query_user_db 拒绝不安全 SQL: %s", sql)
        return "出于安全考虑，仅支持只读查询（SELECT），该请求被拒绝。"

    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + " LIMIT 50"

    try:
        conn = connect_to_database()
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        return f"[SQL 执行错误] {e}"

    if not rows:
        return "查询无结果。"
    result = [dict(r) for r in rows]
    return f"SQL: {sql}\n结果({len(result)}行): " + json.dumps(
        result, ensure_ascii=False, default=str)[:1500]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "当需要最新的、实时的或本地知识库没有的信息时，联网搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_db",
            "description": "查询本地用户数据库（聊天历史 chat_history、用户画像 user_profile）。"
                           "当需要用户自身的历史记录或画像统计时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用自然语言描述要查询的内容"}
                },
                "required": ["question"],
            },
        },
    },
]

_DISPATCH = {
    "web_search": lambda a: web_search(a.get("query", "")),
    "query_user_db": lambda a: query_user_db(a.get("question", "")),
}


def execute_tool(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"未知工具: {name}"
    return fn(args)


if __name__ == "__main__":
    print(web_search("what is retrieval augmented generation"))
    print("safe?", _is_safe_select("SELECT * FROM user_profile"))
    print("safe?", _is_safe_select("DROP TABLE user_profile"))
