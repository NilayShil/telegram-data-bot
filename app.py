import os
import re
import time
import json
import threading
import traceback
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

LOG_FILE = "run.jsonl"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
CHAT_HISTORIES = {}

def log_event(event_type, data):
    log_entry = {
        "timestamp": time.time(),
        "type": event_type,
        "data": data
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def run_python(code: str) -> str:
    """Executes Python code safely and captures stdout or local variables."""
    import sys
    import io

    # Redirect stdout to capture prints
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    local_scope = {}
    try:
        exec_globals = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "requests": __import__("requests"),
            "bs4": __import__("bs4"),
        }
        exec(code, exec_globals, local_scope)
        sys.stdout = old_stdout
        
        printed_val = redirected_output.getvalue()
        if printed_val.strip():
            return printed_val[-8000:]
            
        output = local_scope.get("result", local_scope)
        return str(output)[-8000:]
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error executing code: {str(e)}"

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

def run_agent(chat_id: int, user_message: str) -> dict:
    if chat_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[chat_id] = []

    CHAT_HISTORIES[chat_id].append({"role": "user", "content": user_message})
    CHAT_HISTORIES[chat_id] = CHAT_HISTORIES[chat_id][-20:]

    system_prompt = (
        "You are an expert data analysis bot. Always answer the latest user message.\n"
        "Use the run_python tool whenever you need to download, parse, or compute data.\n"
        "Reply with ONLY the exact raw JSON structure requested in the prompt. "
        "No Markdown formatting, no backticks, no prose.\n"
        "Include a dummy 'log_url' field in your JSON response."
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code to fetch and analyze data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"}
                },
                "required": ["code"]
            }
        }
    }]

    messages = [{"role": "system", "content": system_prompt}] + CHAT_HISTORIES[chat_id]
    start_time = time.time()

    for step in range(10):
        if time.time() - start_time > 210:
            log_event("timeout_warning", "Budget exhausted.")
            break

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        
        # Build dictionary for assistant message to preserve tool calls in history
        assistant_msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg.tool_calls
            ]
        messages.append(assistant_msg_dict)

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                code = args.get("code", "")
                log_event("tool_call", {"code": code})
                
                output = run_python(code)
                log_event("tool_output", {"output": output})
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })
        else:
            final_text = msg.content
            CHAT_HISTORIES[chat_id].append({"role": "assistant", "content": final_text})
            log_event("agent_raw_output", final_text)
            return parse_and_clean_json(final_text)

    return {"answer": "error or timeout", "log_url": f"{BASE_URL}/run.jsonl"}

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
                        except Exception as e:
                            # Print trace to Render Logs for easy debugging
                            print(f"CRITICAL AGENT ERROR: {traceback.format_exc()}")
                            log_event("error", traceback.format_exc())
                            result_json = {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}

                        reply_str = json.dumps(result_json)
                        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
                            "chat_id": chat_id,
                            "text": reply_str
                        })
                        log_event("sent_response", reply_str)
        except Exception as e:
            print(f"POLLING ERROR: {traceback.format_exc()}")
            time.sleep(5)

def keep_warm_worker():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=10)
        except Exception:
            pass

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
