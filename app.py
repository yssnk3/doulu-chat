import os
import json
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY", "")
API_BASE = os.environ.get("API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "")
PERSONA_FILE = os.environ.get("PERSONA_FILE", "persona.md")
MEMORY_FILE = os.environ.get("MEMORY_FILE", "memory.md")

def load_prompt():
    if SYSTEM_PROMPT:
        return SYSTEM_PROMPT
    parts = ["你是逗鹿，不是AI助手。用逗鹿的方式说话和思考。"]
    for f in [PERSONA_FILE, MEMORY_FILE]:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fp:
                parts.append(fp.read())
    return "\n\n".join(parts)

system_prompt = load_prompt()

conversations = {}

# Defer client init to avoid crash on startup if env vars are missing
def get_client():
    return OpenAI(api_key=API_KEY, base_url=API_BASE)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400
    if not API_KEY:
        return jsonify({"error": "API_KEY 环境变量未设置，请在 Railway Variables 中配置"}), 500
    try:
        client = get_client()
    except Exception as e:
        return jsonify({"error": f"OpenAI 客户端初始化失败: {str(e)}"}), 500
    if session_id not in conversations:
        conversations[session_id] = [{"role": "system", "content": system_prompt}]
    conversations[session_id].append({"role": "user", "content": user_msg})
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=conversations[session_id],
            temperature=0.9,
            max_tokens=500,
        )
        reply = resp.choices[0].message.content
        conversations[session_id].append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()
    session_id = data.get("session_id", "default")
    conversations.pop(session_id, None)
    return jsonify({"ok": True})
