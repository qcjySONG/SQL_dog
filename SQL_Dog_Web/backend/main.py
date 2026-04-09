# -*- coding: utf-8 -*-
import sys
import os
import uuid
import time
import threading
import logging
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'sql_dog.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 导入
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 导入配置并更新为相对路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import config
config.LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "sql_dog.log")

# 从环境变量读取 API Keys（不在代码中硬编码）
import os
config.MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")
config.MODELSCOPE_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
config.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
config.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api-inference.modelscope.cn/v1")

config.VALUE_INDICES_PATH = os.path.join(PROJECT_ROOT, "DB_faiss_indices", "mimic_iv", "mimic_iv_columnwise_embeddings_faiss_indices.bin")
config.VALUE_METADATA_PATH = os.path.join(PROJECT_ROOT, "DB_faiss_indices", "mimic_iv", "mimic_iv_columnwise_embeddings_metadata.pkl")
config.KB_INDICES_PATH = os.path.join(PROJECT_ROOT, "DB_faiss_indices", "KB", "KB_columnwise_embeddings_faiss_indices.bin")
config.KB_METADATA_PATH = os.path.join(PROJECT_ROOT, "DB_faiss_indices", "KB", "KB_columnwise_embeddings_metadata.pkl")

app = FastAPI(title="SQL Dog API")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "assets")), name="static")

# 导入后端模块
from agent import run_conversation, set_progress_callback
from conversation_manager import conversation_manager

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations = {}
progress_events = []
progress_lock = threading.Lock()

db_schema = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": ["hadm_id", "subject_id", "admittime", "dischtime", "diagnosis"],
    "prescriptions": ["prescription_id", "hadm_id", "drug", "dose_val_rx", "route"]
}

def generate_conversation_id():
    return str(uuid.uuid4())

def on_progress(step, data=None):
    with progress_lock:
        progress_events.append({"step": step, "data": data or {}, "ts": time.time()})

set_progress_callback(on_progress)

def get_conversation_history(conversation_id):
    if conversation_id not in conversations:
        conv_data = conversation_manager.load_conversation(conversation_id)
        if conv_data:
            conversations[conversation_id] = conv_data
        else:
            conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "turn_count": 0
            }
    return conversations[conversation_id]

@app.get("/")
async def root():
    try:
        index_path = os.path.join(PROJECT_ROOT, "frontend_dist", "index.html")
        return HTMLResponse(open(index_path).read())
    except:
        return {"message": "SQL Dog API running"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/conversation/new")
async def create_conversation():
    conv_id = generate_conversation_id()
    conversations[conv_id] = {
        "history": [],
        "created_at": datetime.now().isoformat(),
        "turn_count": 0
    }
    return {"conversation_id": conv_id}

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id):
    conv_data = get_conversation_history(conversation_id)
    return conv_data

@app.get("/api/conversations")
async def list_conversations():
    try:
        recent_convs = conversation_manager.get_recent_conversations_with_summaries(15)
        return {"conversations": recent_convs}
    except Exception as e:
        return {"conversations": [], "error": str(e)}

@app.get("/api/schema")
async def get_schema():
    return db_schema

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "")
    conversation_id = body.get("conversation_id", "")
    
    logger.info(f"收到消息: {message[:50]} | conversation_id: {conversation_id}")
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    if not conversation_id:
        conversation_id = generate_conversation_id()
        conversations[conversation_id] = {
            "history": [],
            "created_at": datetime.now().isoformat(),
            "turn_count": 0
        }
    
    conv_data = get_conversation_history(conversation_id)
    
    with progress_lock:
        progress_events.clear()
    
    result_holder = {"result": None, "done": False, "error": None}
    timestamps = {"intent_done": None, "table_done": None, "sql_done": None}
    seen_steps = set()
    t_start = time.time()
    
    def run_in_thread():
        try:
            logger.info(f"开始处理对话: {conversation_id}")
            result_holder["result"] = run_conversation(
                question=message,
                conversation_id=conversation_id,
                history=conv_data["history"],
                progress_callback=on_progress
            )
            result_holder["done"] = True
            logger.info(f"对话处理完成: {conversation_id}")
        except Exception as e:
            logger.error(f"对话处理错误: {str(e)}")
            result_holder["error"] = str(e)
            result_holder["done"] = True
    
    t = threading.Thread(target=run_in_thread)
    t.start()
    
    progress_data = []
    
    while not result_holder["done"]:
        with progress_lock:
            events = list(progress_events)
        for evt in events:
            current_step = evt["step"]
            current_data = evt["data"]
            
            if current_step not in seen_steps:
                seen_steps.add(current_step)
                ts = current_data.get("timestamp", evt["ts"])
                timestamps[current_step] = ts
                
                step_info = {"step": current_step, "timestamp": ts, "data": current_data}
                progress_data.append(step_info)
        
        time.sleep(0.05)
    
    t.join()
    
    if result_holder["error"]:
        return JSONResponse(status_code=500, content={"error": result_holder["error"], "conversation_id": conversation_id})
    
    result = result_holder["result"]
    conv_data["history"] = result.get("history", [])
    conv_data["turn_count"] = conv_data.get("turn_count", 0) + 1
    conversation_manager.save_conversation(conversation_id, conv_data)
    
    intent_ts = timestamps.get("intent_done") or t_start
    table_ts = timestamps.get("table_done") or intent_ts
    sql_ts = timestamps.get("sql_done") or table_ts
    
    logger.info(f"返回结果: answer={result.get('final_answer', '')[:100]}, sql={result.get('generated_sql', '')[:50]}")
    
    return {
        "conversation_id": conversation_id,
        "answer": result.get("final_answer", "") or result.get("answer", ""),
        "sql": result.get("generated_sql", ""),
        "result": result.get("sql_result"),
        "intent": result.get("step_details", {}).get("intent", "query"),
        "selected_tables": result.get("selected_tables", []),
        "history": conv_data["history"],
        "progress": progress_data,
        "timings": {
            "intent": round(intent_ts - t_start, 1),
            "table": round(table_ts - intent_ts, 1),
            "sql": round(sql_ts - table_ts, 1),
            "total": round(sql_ts - t_start, 1)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7863)
