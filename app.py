import os
import re
import time
import json
import threading
import traceback
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI()

LOG_FILE = "run.jsonl"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# 1. Logging Helper
def log_event(event_type, data):
    log_entry = {
        "timestamp": time.time(),
        "type": event_type,
        "data": data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# 2. Python Tool (Passed directly to Gemini for Automatic Function Calling)
def run_python(code: str) -> str:
    """Executes Python code to download, parse, or analyze data and returns the result."""
    log_event("tool_call", {"code": code})
    local_scope = {}
    try:
        exec_globals = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "requests": __import__("requests"),
            "bs4": __import__("bs4"),
        }
        exec(code, exec_globals, local_scope)
        output = local_scope.get("result", local_scope)
        result_str = str(output)[-8000:]
        log_event("tool_output", {"output": result_str})
        return result_str
    except Exception as e:
        err_msg = f"Error executing code: {str(e)}"
        log_event("tool_output", {"output": err_msg})
        return err_msg

# 3. JSON Sanitizer
def parse_and_clean_json(raw_text: str) -> dict:
    log_url = f"{BASE_URL}/run.jsonl"
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
            
        data = json.loads(cleaned)
        data["log_url"] = log_url
        if "answer" not in data:
            data = {"answer": data, "log_url": log_url}
        return data
    except Exception:
        return {"answer": raw_text, "log_url": log_url}

# 4. Agent Execution with Built-in Tool Support
def run_agent(user_message: str) -> dict:
    prompt = (
        "You are an expert data analyst bot.\n"
        "Use the run_python tool whenever you need to download, parse, or compute data.\n"
        "Reply with ONLY the exact raw JSON structure requested in the user prompt. "
        "No Markdown formatting, no backticks, no extra prose.\n"
        "Include a dummy 'log_url' field in your JSON response.\n\n"
        f"User Query: {user_message}"
    )

    try:
        # Pass Python callable directly; SDK automatically runs the code and feeds output back
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "tools": [run_python],
                "temperature": 0.1
            }
        )
        final_text = response.text or ""
        log_event("agent_raw_output", final_text)
        return parse_and_clean_json(final_text)
    except Exception as e:
        print("GENAI ERROR:", traceback.format_exc())
        log_event("genai_error", traceback.format_exc())
        return {"answer": "error in agent execution", "log_url": f"{BASE_URL}/run.jsonl"}

# 5. Telegram Worker Loop
def telegram_polling_worker():
    offset = 0
    while True:
        try:
            url = f"{TELEGRAM_API_URL}/getUpdates?offset={offset}&timeout=30"
            res = requests.get(url, timeout=35).json()
            
            if "result" in res:
                for update in res["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        text = update["message"]["text"]
                        
                        log_event("incoming_message", {"chat_id": chat_id, "text": text})
                        
                        try:
                            result_json = run_agent(text)
                        except Exception:
                            print("AGENT CRASH:", traceback.format_exc())
                            log_event("error", traceback.format_exc())
                            result_json = {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}

                        reply_str = json.dumps(result_json)
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
                            "chat_id": chat_id,
                            "text": reply_str
                        })
                        log_event("sent_response", reply_str)
        except Exception:
            time.sleep(5)

# 6. Keep-Alive Worker
def keep_warm_worker():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=10)
        except Exception:
            pass

# 7. FastAPI Endpoints
@app.get("/health")
def health():
    return {"ok": True, "status": "running"}

@app.get("/run.jsonl")
def get_logs():
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="application/x-jsonlines")
    return JSONResponse(content={"error": "Log file not created yet"}, status_code=404)

@app.on_event("startup")
def start_background_tasks():
    threading.Thread(target=telegram_polling_worker, daemon=True).start()
    threading.Thread(target=keep_warm_worker, daemon=True).start()
