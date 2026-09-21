import os
import re
import time
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
import numpy as np

# โหลดฟังก์ชันค้นหาจากไฟล์เดิม
from retriever import load_chunks, load_qa, load_embeddings, embeddings_ready, search, search_pdf
from prompt import PROMPT_PYBOT

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")
try:
    client = genai.Client(api_key=api_key)
except ValueError:
    client = None

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

CHAT_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=2048,
    system_instruction=PROMPT_PYBOT,
    safety_settings=SAFETY_SETTINGS,
)

BASE_DIR = os.path.dirname(__file__)
MAX_HISTORY_MESSAGES = 6

def normalize_query(query: str) -> str:
    if not client: return query
    try:
        result = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=(
                f"คำถามเกี่ยวกับ Python: '{query}'\n"
                f"งาน: แก้ typo, ดึง Python keyword ที่ถูกต้อง, และแปลงเป็นคำภาษาไทยที่เกี่ยวข้อง\n"
                f"ตอบเฉพาะ keywords 2-5 คำ คั่นด้วย space ห้ามอธิบาย"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30),
        )
        return f"{query} {result.text.strip().replace(chr(10), ' ')}"
    except Exception:
        return query

def expand_query(query: str, chunks: list) -> str:
    if not client: return query
    try:
        initial_context = search_pdf(query, chunks)
        result = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=(
                f"เนื้อหาจากหนังสือ Python MSU:\n{initial_context[:1500]}\n\n"
                f"คำถาม: {query}\n\n"
                f"ดึง keywords ภาษาไทยและอังกฤษที่เกี่ยวข้องกับ Python concept นี้ 2-3 คำ\n"
                f"ตอบเฉพาะ keywords คั่นด้วย space เท่านั้น ห้ามอธิบาย"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=20),
        )
        return f"{query} {result.text.strip().replace(chr(10), ' ')}"
    except Exception:
        return query

GREETING_PATTERNS = {
    "สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ", "hello", "hi",
    "ขอบคุณ", "thanks", "ทำอะไรได้", "ช่วยอะไรได้",
}
def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    if len(t) >= 30:
        return False
    for g in GREETING_PATTERNS:
        if g.isascii():
            if re.search(rf"\b{re.escape(g)}\b", t):
                return True
        elif g in t:
            return True
    return False

# Cache เก็บข้อมูลไว้ในหน่วยความจำ
DATA_CACHE = {}

def load_all_data():
    if "chunks" not in DATA_CACHE:
        DATA_CACHE["chunks"] = load_chunks(os.path.join(BASE_DIR, "python_data.md"))
        DATA_CACHE["qa"] = load_qa(os.path.join(BASE_DIR, "python_qa.xlsx"))
        DATA_CACHE["embeddings"] = load_embeddings(os.path.join(BASE_DIR, "embeddings.npy"))
    return DATA_CACHE["chunks"], DATA_CACHE["qa"], DATA_CACHE["embeddings"]

def embed_query(text: str) -> np.ndarray:
    if not client: return None
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text[:2000],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    chunks, qa, emb = load_all_data()
    
    await cl.Message(
        content="สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ 🐍\n\n(ขับเคลื่อนด้วย Chainlit + Gemini)"
    ).send()
    
    if emb is None:
        await cl.Message(content="ℹ️ คำเตือน: ไม่มีไฟล์ embeddings.npy ระบบจะใช้เฉพาะ Keyword Search").send()
    elif not embeddings_ready(emb, chunks):
        await cl.Message(content="⚠️ คำเตือน: embeddings.npy ไม่ตรงกับเนื้อหา ระบบจะปิด Semantic Search ชั่วคราว").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    chunks, qa_rows, embeddings = load_all_data()
    
    if is_greeting(user_input):
        greeting_reply = "สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ ถามเรื่อง Python ได้เลยครับ 🐍"
        await cl.Message(content=greeting_reply).send()
        return

    # แสดง Step การค้นหา (ผู้ใช้กดดูรายละเอียดได้)
    async with cl.Step(name="🔍 กำลังค้นหาข้อมูลในหนังสือ...") as step:
        step.input = user_input
        normalized = normalize_query(user_input)
        expanded = expand_query(normalized, chunks)
        relevant_context = search(expanded, chunks, qa_rows, embeddings=embeddings, embed_fn=embed_query)
        step.output = relevant_context
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # โหลดประวัติแบบ dict กลางไว้ก่อน
    session_messages = cl.user_session.get("messages")
    raw_history = []
    for msg in session_messages[-MAX_HISTORY_MESSAGES:]:
        raw_history.append({"role": msg["role"], "content": msg["content"]})
        
    # แทรก Context แบบ dict
    raw_history.append({
        "role": "user",
        "content": (
            f"[เนื้อหาจากหนังสือ Python MSU ที่เกี่ยวข้องกับคำถามถัดไป — ใช้เนื้อหานี้เป็นหลักในการตอบ]\n\n"
            f"{relevant_context}\n\n"
            f"[สิ้นสุดเนื้อหาจากหนังสือ]"
        )
    })
    raw_history.append({
        "role": "model",
        "content": "รับทราบครับ จะตอบโดยอ้างอิงจากเนื้อหาหนังสือที่ให้มาเป็นหลักครับ"
    })
    
    # เตรียมส่งข้อความแบบสตรีม
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        success = False
        if openrouter_key:
            # === โหมด OpenRouter ===
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )

            # แปลง role: "model" -> "assistant"
            or_history = [{"role": "system", "content": PROMPT_PYBOT}]
            for h in raw_history:
                role = "assistant" if h["role"] == "model" else h["role"]
                or_history.append({"role": role, "content": h["content"]})
            or_history.append({"role": "user", "content": user_input})

            # ใช้โมเดลเดิมแต่ผ่าน OpenRouter (สามารถเปลี่ยนเป็น claude-3.5-sonnet ได้ตามต้องการ)
            models = ["google/gemini-2.5-flash", "google/gemini-flash-1.5", "openai/gpt-4o-mini"]
            for model_name in models:
                try:
                    stream = await openai_client.chat.completions.create(
                        model=model_name,
                        messages=or_history,
                        temperature=0.0,
                        max_tokens=2048,
                        stream=True
                    )

                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            await msg.stream_token(chunk.choices[0].delta.content)

                    await msg.update()
                    success = True
                    break
                except Exception as e:
                    print(f"OpenRouter Fallback ({model_name} failed): {e}")
                    continue
        else:
            # === โหมด Google Gemini ดั้งเดิม ===
            if not client:
                await cl.Message(content="❌ ไม่พบ Gemini API Key หรือ OpenRouter API Key").send()
                return

            gemini_history = []
            for h in raw_history:
                gemini_history.append(types.Content(role=h["role"], parts=[types.Part(text=h["content"])]))

            models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for model in models:
                for attempt in range(3):
                    try:
                        chat = client.chats.create(model=model, config=CHAT_CONFIG, history=gemini_history)
                        response_stream = chat.send_message_stream(user_input)

                        for chunk in response_stream:
                            if chunk.text:
                                await msg.stream_token(chunk.text)

                        await msg.update()
                        success = True
                        break
                    except errors.ServerError:
                        if attempt < 2:
                            await asyncio.sleep(7)
                        continue
                    except Exception as e:
                        if "404" in str(e) or "not found" in str(e).lower():
                            break  # ลองเปลี่ยนไปใช้ model ถัดไปใน list
                        else:
                            raise e
                if success:
                    break

        if not success:
            await cl.Message(content="⚠️ ไม่สามารถเชื่อมต่อกับโมเดลใดๆ ได้ในขณะนี้ กรุณาตรวจสอบ API Key หรือลองใหม่ภายหลัง").send()
        else:
            # บันทึกประวัติ
            session_messages.append({"role": "user", "content": user_input})
            session_messages.append({"role": "model", "content": msg.content})
            cl.user_session.set("messages", session_messages)

    except Exception as e:
        await cl.Message(content=f"❌ เกิดข้อผิดพลาดจาก API: {str(e)}").send()
