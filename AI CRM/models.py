from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL, logger

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


def chat(messages, model: str = None):
    client = get_client()
    response = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_once(messages, model: str = None, temperature: float = 0.0) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return resp.choices[0].message.content or ""


def chat_with_tools(messages: List[Dict], tools: List[Dict],
                    tool_executor: Callable[[str, dict], str],
                    model: str = None, max_rounds: int = 4) -> str:
    client = get_client()
    msgs = list(messages)

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model or CHAT_MODEL,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return msg.content or ""

        msgs.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            logger.info("tool call: %s(%s)", name, args)
            try:
                result = tool_executor(name, args)
            except Exception as e: 
                result = f"工具执行失败: {e}"
                logger.exception("tool %s failed", name)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    final = client.chat.completions.create(
        model=model or CHAT_MODEL, messages=msgs, stream=False,
    )
    return final.choices[0].message.content or ""


if __name__ == "__main__":
    for d in chat([{"role": "user", "content": "用一句话介绍你自己"}]):
        print(d, end="", flush=True)
    print()
