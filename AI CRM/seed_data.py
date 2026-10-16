from config import STORE_MARKET, STORE_COURSE, logger
from vectorstore import get_store

MARKET_DOCS = [
    "AI 工程师是当前需求量最大的技术岗位之一，涵盖机器学习工程师、算法工程师、数据科学家、AIGC 应用工程师等方向。",
    "转行做 AI 通常需要掌握 Python 编程、数学基础（线性代数、概率统计）、机器学习与深度学习框架（PyTorch/TensorFlow）。",
    "一线城市 AI 算法工程师应届起薪普遍在 20-35 万/年，有 3-5 年经验的资深工程师年薪可达 50-100 万。",
    "非科班背景转 AI 的可行路径：先补足编程与数学基础，再通过项目实战积累作品集，最后针对性投递岗位。",
    "AIGC 与大模型应用是近两年增长最快的方向，Prompt 工程、RAG、Agent 开发等技能需求旺盛。",
]

COURSE_DOCS = [
    "《AI 工程师全栈实战课》包含 Python 基础、机器学习、深度学习、大模型应用四大模块，共 120 课时。",
    "课程采用直播+录播+项目实战的模式，配备助教答疑，学习周期约 4-6 个月。",
    "课程原价 6980 元，新学员首月报名享 8 折优惠，老学员推荐可再减 500 元。",
    "报名方式：在官网或小程序填写手机号提交，顾问会在 1 个工作日内联系确认班期。",
    "结课项目包括一个完整的 RAG 智能问答系统和一个 Agent 应用，可作为求职作品集。",
    "师资均为一线大厂算法专家，具备 5 年以上实战与教学经验。",
]


def main():
    m = get_store(STORE_MARKET)
    c = get_store(STORE_COURSE)
    m.add(MARKET_DOCS, [{"source": "market_kb"} for _ in MARKET_DOCS])
    c.add(COURSE_DOCS, [{"source": "course_kb"} for _ in COURSE_DOCS])
    logger.info("seed 完成：market=%d 条, course=%d 条", m.count(), c.count())
    print(f"seed 完成：market={m.count()} 条, course={c.count()} 条")


if __name__ == "__main__":
    main()
