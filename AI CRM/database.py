from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import List, Dict, Optional

from config import DB_PATH, logger

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def connect_to_database() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")  
        _init_schema(conn)
        _conn = conn
        logger.info("database: 已连接 %s", DB_PATH)
        return _conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            ts           REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_phone ON chat_history(phone_number);

        CREATE TABLE IF NOT EXISTS user_profile (
            phone_number   TEXT PRIMARY KEY,
            name           TEXT,
            intent_history TEXT DEFAULT '[]',   -- JSON 数组，记录历史意图编号
            tags           TEXT DEFAULT '[]',   -- JSON 数组，画像标签
            notes          TEXT DEFAULT '',
            created_at     REAL,
            updated_at     REAL
        );
        """
    )
    conn.commit()


def fetch_chat_history(phone_number: str, limit: int = 20) -> List[Dict]:
    conn = connect_to_database()
    with _lock:
        rows = conn.execute(
            "SELECT role, content FROM chat_history "
            "WHERE phone_number=? ORDER BY id DESC LIMIT ?",
            (phone_number, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def initialize_chat_history(phone_number: str):
    conn = connect_to_database()
    now = time.time()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO user_profile "
            "(phone_number, intent_history, tags, notes, created_at, updated_at) "
            "VALUES (?, '[]', '[]', '', ?, ?)",
            (phone_number, now, now),
        )
        conn.commit()
    logger.info("database: 初始化新用户 %s", phone_number)


def append_message(phone_number: str, role: str, content: str):
    conn = connect_to_database()
    with _lock:
        conn.execute(
            "INSERT INTO chat_history (phone_number, role, content, ts) "
            "VALUES (?, ?, ?, ?)",
            (phone_number, role, content, time.time()),
        )
        conn.commit()


def update_chat_history(phone_number: str, user_query: str, assistant_response: str):
    append_message(phone_number, "user", user_query)
    append_message(phone_number, "assistant", assistant_response)


def get_user_profile(phone_number: str) -> Optional[Dict]:
    conn = connect_to_database()
    with _lock:
        row = conn.execute(
            "SELECT * FROM user_profile WHERE phone_number=?", (phone_number,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["intent_history"] = json.loads(d.get("intent_history") or "[]")
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


def update_user_profile(phone_number: str, *, intent_code: Optional[int] = None,
                        name: Optional[str] = None, add_tags: Optional[List[str]] = None,
                        notes: Optional[str] = None):
    conn = connect_to_database()
    prof = get_user_profile(phone_number)
    now = time.time()
    if prof is None:
        initialize_chat_history(phone_number)
        prof = get_user_profile(phone_number)

    intent_history = prof["intent_history"]
    if intent_code is not None:
        intent_history.append(intent_code)
        intent_history = intent_history[-50:]  
    tags = prof["tags"]
    if add_tags:
        for t in add_tags:
            if t not in tags:
                tags.append(t)

    with _lock:
        conn.execute(
            "UPDATE user_profile SET name=COALESCE(?, name), "
            "intent_history=?, tags=?, notes=COALESCE(?, notes), updated_at=? "
            "WHERE phone_number=?",
            (name, json.dumps(intent_history, ensure_ascii=False),
             json.dumps(tags, ensure_ascii=False), notes, now, phone_number),
        )
        conn.commit()


if __name__ == "__main__":
    phone = "13800000000"
    initialize_chat_history(phone)
    update_chat_history(phone, "你好", "您好，有什么可以帮您？")
    update_user_profile(phone, intent_code=1, add_tags=["AI求职"])
    print("history:", fetch_chat_history(phone))
    print("profile:", get_user_profile(phone))
