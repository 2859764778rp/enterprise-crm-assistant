from __future__ import annotations

import re

from config import INTENT_MODEL, INTENT_MARKET, INTENT_COURSE, INTENT_OTHER, logger
from models import chat_once

_SYSTEM = (
    "你是意图分类器。阅读用户输入，判断其属于以下哪一类，只回复对应的数字，不要解释：\n"
    "1 = AI 职业咨询（如就业前景、岗位、薪资、职业规划、转行 AI 等）\n"
    "2 = 课程咨询（如课程内容、大纲、价格、报名、学习路径、师资等）\n"
    "0 = 其他（闲聊、常识、时事、与上面两类都无关的问题）\n"
    "只输出 0、1 或 2 中的一个数字。"
)

_VALID = {0: INTENT_OTHER, 1: INTENT_MARKET, 2: INTENT_COURSE}


def detect_intent_code(query: str) -> int:
    try:
        text = chat_once(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": query}],
            model=INTENT_MODEL,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("intent: 模型调用失败，回退为 OTHER。原因: %s", e)
        return INTENT_OTHER

    m = re.search(r"[012]", text or "")
    code = int(m.group()) if m else 0
    logger.info("intent: query=%r -> code=%s (raw=%r)", query, code, text)
    return _VALID.get(code, INTENT_OTHER)


if __name__ == "__main__":
    for q in ["转行做AI工程师前景怎么样", "你们的深度学习课程多少钱", "今天几号"]:
        print(q, "->", detect_intent_code(q))
