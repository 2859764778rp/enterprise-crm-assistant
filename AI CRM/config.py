import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VEC_DIR = DATA_DIR / "vecstore"        
DB_PATH = DATA_DIR / "crm.db"         
LOG_PATH = DATA_DIR / "app.log"

for _d in (DATA_DIR, VEC_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
INTENT_MODEL = os.getenv("INTENT_MODEL", CHAT_MODEL)

INTENT_MARKET = 1   
INTENT_COURSE = 2   
INTENT_OTHER = 0    

STORE_MARKET = "market"
STORE_COURSE = "course"

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )
    return logging.getLogger("ai_crm")



logger = configure_logging()
