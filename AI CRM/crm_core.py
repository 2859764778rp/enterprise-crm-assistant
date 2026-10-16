from __future__ import annotations

from typing import Iterator

from config import (logger, INTENT_MARKET, INTENT_COURSE,
                    STORE_MARKET, STORE_COURSE, RAG_TOP_K)
from database import (connect_to_database, fetch_chat_history,
                      initialize_chat_history, update_chat_history,
                      get_user_profile, update_user_profile)
from intent import detect_intent_code
from vectorstore import get_store
from models import chat, chat_with_tools
from tools import TOOL_SCHEMAS, execute_tool

INITIAL_GREETING = (
    "您好，我是 AI CRM 智能助手 🤖。我可以为您提供 AI 职业咨询、课程咨询，"
    "也能帮您解答其他问题。请问有什么可以帮到您？"
)

_BASE_SYSTEM = "你是一家 AI 教育公司的专业智能客服，回答准确、简洁、友好，使用中文。"


def _profile_strategy(profile) -> str:
    if not profile:
        return "这是新用户，请给出友好、完整的引导性回答。"
    hints = []
    if profile.get("name"):
        hints.append(f"用户称呼：{profile['name']}")
    if profile.get("tags"):
        hints.append("用户画像标签：" + "、".join(profile["tags"]))
    hist = profile.get("intent_history") or []
    if hist:
        if hist.count(INTENT_COURSE) >= 2:
            hints.append("该用户多次咨询课程，可适当推荐并给出报名引导。")
        if hist.count(INTENT_MARKET) >= 2:
            hints.append("该用户关注职业发展，可结合就业前景作答。")
    return "；".join(hints) if hints else "老用户，请结合其历史保持连贯。"


def _build_messages(system_extra, history, query):
    msgs = [{"role": "system", "content": _BASE_SYSTEM + "\n" + system_extra}]
    msgs.extend(history[-10:])  
    msgs.append({"role": "user", "content": query})
    return msgs


def _rag_context(store_name: str, query: str) -> str:
    hits = get_store(store_name).query(query, top_k=RAG_TOP_K)
    if not hits:
        return ""
    ctx = "\n".join(f"[知识{i+1}] {h['text']}" for i, h in enumerate(hits))
    logger.info("RAG[%s]: 命中 %d 条", store_name, len(hits))
    return ctx


def interview_stream(phone_number: str, query: str) -> Iterator[str]:
    connect_to_database()
    history = fetch_chat_history(phone_number)

    if not history:
        initialize_chat_history(phone_number)
        update_chat_history(phone_number, query, INITIAL_GREETING)
        yield INITIAL_GREETING
        return

    intent_code = detect_intent_code(query)
    profile = get_user_profile(phone_number)
    strategy = _profile_strategy(profile)

    response_parts = []

    if intent_code == INTENT_MARKET:
        ctx = _rag_context(STORE_MARKET, query)
        sys_extra = (f"回答策略：{strategy}\n"
                     f"以下是「AI 职业咨询」知识库检索结果，请优先据此作答；"
                     f"若不足可结合常识：\n{ctx}")
        for delta in chat(_build_messages(sys_extra, history, query)):
            response_parts.append(delta)
            yield delta

    elif intent_code == INTENT_COURSE:
        ctx = _rag_context(STORE_COURSE, query)
        sys_extra = (f"回答策略：{strategy}\n"
                     f"以下是「课程咨询」知识库检索结果，请优先据此作答：\n{ctx}")
        for delta in chat(_build_messages(sys_extra, history, query)):
            response_parts.append(delta)
            yield delta

    else:
        full = _handle_other(phone_number, query, history, strategy)
        response_parts.append(full)
        yield full

    response = "".join(response_parts)
    update_chat_history(phone_number, query, response)
    update_user_profile(phone_number, intent_code=intent_code)


def _handle_other(phone_number, query, history, strategy) -> str:
    sys_extra = (f"回答策略：{strategy}\n"
                 f"如果问题涉及实时信息可调用 web_search；"
                 f"若涉及该用户历史或画像可调用 query_user_db。当前用户手机号：{phone_number}")
    msgs = _build_messages(sys_extra, history, query)
    return chat_with_tools(msgs, TOOL_SCHEMAS, execute_tool)


def SeaChatInterview(phone_number: str, query: str) -> str:
    return "".join(interview_stream(phone_number, query))


if __name__ == "__main__":
    phone = "13900000001"
    print("A:", SeaChatInterview(phone, "你好"))
    print("B:", end=" ")
    for d in interview_stream(phone, "AI 工程师就业前景怎么样"):
        print(d, end="", flush=True)
    print()
