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
from google.genai import types
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
CHAT_HISTORIES = {}

# 1. Logging Helper
def log_event(event_type, data):
    log_entry = {
        "timestamp": time.time(),
        "type": event_type,
        "data": data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# 2. Safe Python Execution Tool
def run_python(code: str) -> str:
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
        return str(output)[-8000:]
    except Exception as e:
        return f"Error executing code: {str(e)}"

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

# 4. Gemini Agent Loop
def run_agent(chat_id: int, user_message: str) -> dict:
    if chat_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[chat_id] = []

    CHAT_HISTORIES[chat_id].append({"role": "user", "parts": [{"text": user_message}]})
    CHAT_HISTORIES[chat_id] = CHAT_HISTORIES[chat_id][-20:]

    system_instruction = (
        "You are an expert data analysis bot. Always answer the latest user message.\n"
        "Use the run_python tool whenever you need to download, parse, or compute data.\n"
        "Reply with ONLY the exact raw JSON structure requested in the prompt. "
        "No Markdown formatting, no backticks, no prose.\n"
        "Include a dummy 'log_url' field in your JSON response."
    )

    # Define tool for Gemini
    python_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="run_python",
                description="Execute Python code to fetch and analyze data.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "code": types.Schema(
                            type="STRING",
                            description="Python code block to execute"
                        )
                    },
                    required=["code"]
                )
            )
        ]
    )

    contents = CHAT_HISTORIES[chat_id].copy()
    start_time = time.time()

    for step in range(10):
        if time.time() - start_time > 210:  # Timeout safety budget
            log_event("timeout_warning", "Budget exhausted.")
            break

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[python_tool],
            temperature=0.1
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Fast & reliable frontier model
            contents=contents,
            config=config
        )

        # Check for function/tool calls
        if response.function_calls:
            for call in response.function_calls:
                fn_name = call.name
                fn_args = call.args
                
                if fn_name == "run_python":
                    code = fn_args.get("code", "")
                    log_event("tool_call", {"code": code})
                    
                    tool_output = run_python(code)
                    log_event("tool_output", {"output": tool_output})
                    
                    # Append model response & function output to context
                    contents.append(response.candidates[0].content)
                    contents.append(types.Part.from_function_response(
                        name="run_python",
                        response={"result": tool_output}
                    ))
        else:
            final_text = response.text
            CHAT_HISTORIES[chat_id].append({"role": "model", "parts": [{"text": final_text}]})
            log_event("agent_raw_output", final_text)
            return parse_and_clean_json(final_text)

    return {"answer": "error or timeout", "log_url": f"{BASE_URL}/run.jsonl"}

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
                            result_json = run_agent(chat_id, text)
                        except Exception:
                            # Print traceback in Render logs for easy debugging
                            print(traceback.format_exc())
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

# 6. Keep Alive Loop
def keep_warm_worker():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=10)
        except Exception:
            pass

# 7. FastAPI Routes & Startup
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
