"""
配置文件：包括API密钥、模型参数等。
"""
import os
from dotenv import load_dotenv

# 加载环境变量（如果存在.env文件）
load_dotenv()

# ==============================================
# API 配置（从环境变量读取，不在代码中硬编码）
# ==============================================

# ModelScope API 配置
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
MODELSCOPE_MODEL = os.getenv("MODELSCOPE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# DeepSeek API 配置（也可用 ModelScope）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api-inference.modelscope.cn/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# Embedding 模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api-inference.modelscope.cn/v1")

# ==============================================
# LLM 角色分离配置
# ==============================================

# 1. 意图识别LLM (INTENT_LLM)
INTENT_LLM_MODEL = MODELSCOPE_MODEL
INTENT_LLM_API_KEY = MODELSCOPE_API_KEY
INTENT_LLM_BASE_URL = MODELSCOPE_BASE_URL

# 2. 表选择LLM (TABLE_LLM)
TABLE_LLM_MODEL = MODELSCOPE_MODEL
TABLE_LLM_API_KEY = MODELSCOPE_API_KEY
TABLE_LLM_BASE_URL = MODELSCOPE_BASE_URL
TABLE_SELECTION_N = 1

# 3. SQL生成LLM (SQL_LLM)
SQL_LLM_MODEL = DEEPSEEK_MODEL
SQL_LLM_API_KEY = DEEPSEEK_API_KEY
SQL_LLM_BASE_URL = DEEPSEEK_BASE_URL

# ==============================================
# 数据库配置
# ==============================================
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "DB", "mimic_iv.sqlite")
DB_PATH = os.path.abspath(DB_PATH)

DDL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "ddl_cache")

# ==============================================
# 对话配置
# ==============================================
MAX_TURNS = 10
MAX_SQL_RETRIES = 3

# ==============================================
# 检索路由配置
# ==============================================
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "mytoken")
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)

# 值检索路径（从环境变量或使用默认相对路径）
VALUE_INDICES_PATH = os.getenv("VALUE_INDICES_PATH", os.path.join(os.path.dirname(__file__), "..", "DB_faiss_indices", "mimic_iv", "mimic_iv_columnwise_embeddings_faiss_indices.bin"))
VALUE_METADATA_PATH = os.getenv("VALUE_METADATA_PATH", os.path.join(os.path.dirname(__file__), "..", "DB_faiss_indices", "mimic_iv", "mimic_iv_columnwise_embeddings_metadata.pkl"))

# 知识库检索路径
KB_INDICES_PATH = os.getenv("KB_INDICES_PATH", os.path.join(os.path.dirname(__file__), "..", "DB_faiss_indices", "KB", "KB_columnwise_embeddings_faiss_indices.bin"))
KB_METADATA_PATH = os.getenv("KB_METADATA_PATH", os.path.join(os.path.dirname(__file__), "..", "DB_faiss_indices", "KB", "KB_columnwise_embeddings_metadata.pkl"))

# 日志配置
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "sql_dog.log")
