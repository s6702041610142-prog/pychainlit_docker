import os
import re
import time
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np

# โหลดฟังก์ชันค้นหาจากไฟล์เดิม
from retriever import load_chunks, load_qa, load_embeddings, embeddings_ready, search, search_pdf
from prompt import PROMPT_PYBOT

load_dotenv()

# log ของแอปเอง — ใช้ logging แทน print() ให้ขึ้นใน log ของ Render/Streamlit Cloud ครบทุกบรรทัด
log = logging.getLogger("pybot")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - pybot - %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False
api_key = os.getenv("GEMINI_API_KEY", "")
try:
    # timeout (ms) กันคำขอที่ค้างไม่ตอบ และปิดการ retry ภายใน SDK (attempts=1)
    # — เดิม SDK แอบลองซ้ำเองตอนโดน 429/503 ทำให้ผู้ใช้รอเกือบ 2 นาทีก่อนสลับโมเดล
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
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
MAX_HISTORY_MESSAGES = 20   # 10 รอบคุย — 6 (3 รอบ) ทำให้ลืมสิ่งที่ผู้ใช้บอกเร็วเกินไป

# ขั้นช่วยค้นหา (แก้คำผิด / ขยายคำค้น / embedding) ไม่จำเป็นต้องสำเร็จ — ให้เวลาสั้นแล้วข้ามไป (API รับขั้นต่ำ 10 วินาที)
HELPER_MODEL = "gemini-3.5-flash-lite"
HELPER_HTTP = types.HttpOptions(timeout=10_000, retry_options=types.HttpRetryOptions(attempts=1))

# โมเดลที่เพิ่งโควตาหมดหรือเซิร์ฟเวอร์ล่ม ข้ามไปชั่วคราว ไม่ต้องเสียเวลาลองซ้ำทุกคำถาม
QUOTA_COOLDOWN = 60   # วินาที
_quota_until: dict[str, float] = {}

def _cooling(model: str) -> bool:
    return time.monotonic() < _quota_until.get(model, 0.0)

def _note_failure(where: str, model: str, e: Exception):
    msg = str(e)
    # 429/402 = โควตาหมด, 503/504 = เซิร์ฟเวอร์ Gemini คนใช้เยอะ/ตอบไม่ทัน — ทั้งคู่มักเป็นต่อเนื่องสักพัก
    if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "402", "503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED")) \
            or isinstance(e, asyncio.TimeoutError):
        _quota_until[model] = time.monotonic() + QUOTA_COOLDOWN
    log.warning(f"{where} ({model} failed): {type(e).__name__} {msg[:160]}")

# ขั้น rewrite เป็นแค่ตัวช่วย — รอไม่เกินเท่านี้ ถ้า Gemini ช้ากว่านี้ก็ค้นด้วยคำถามเดิมไปเลย
# (timeout ของ API ตั้งต่ำกว่า 10 วินาทีไม่ได้ จึงตัดรอเองฝั่งเรา คำขอที่ค้างจะจบเองในพื้นหลัง)
REWRITE_WAIT = 3   # วินาที
_rewrite_pool = ThreadPoolExecutor(max_workers=4)

def rewrite_query(query: str, chunks: list) -> str:
    """แก้คำสะกดผิด + เติม keyword ไทย/อังกฤษ ในการเรียก Gemini ครั้งเดียว
    (เดิมแยกเป็น normalize_query + expand_query สองครั้งต่อกัน ทำให้รอนานก่อนเริ่มค้น)"""
    if not client or _cooling(HELPER_MODEL): return query
    future = _rewrite_pool.submit(_rewrite_query, query, chunks)
    try:
        return future.result(timeout=REWRITE_WAIT)
    except FutureTimeout:
        log.info(f"rewrite_query skipped: slower than {REWRITE_WAIT}s")
        return query

def _rewrite_query(query: str, chunks: list) -> str:
    try:
        initial_context = search_pdf(query, chunks)   # keyword search ในเครื่อง ไม่เรียก API
        result = client.models.generate_content(
            model=HELPER_MODEL,
            contents=(
                f"เนื้อหาจากหนังสือ Python MSU:\n{initial_context[:1500]}\n\n"
                f"คำถามเกี่ยวกับ Python: '{query}'\n\n"
                f"งาน: แก้คำที่สะกดผิด แล้วบอก Python keyword ที่ถูกต้อง และคำภาษาไทย/อังกฤษที่เกี่ยวข้อง 3-6 คำ\n"
                f"ตอบเฉพาะ keywords คั่นด้วย space เท่านั้น ห้ามอธิบาย"
            ),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=40, http_options=HELPER_HTTP),
        )
        return f"{query} {result.text.strip().replace(chr(10), ' ')}"
    except Exception as e:
        _note_failure("rewrite_query skipped", HELPER_MODEL, e)
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
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", http_options=HELPER_HTTP),
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)

# แคชผล self-test ของ semantic search ไว้ระดับ process — ไม่ยิง API ซ้ำทุกครั้งที่มีคนเปิดแชทใหม่
# ผลสำเร็จแคชถาวร แต่ผลล้มเหลวแคชแค่ช่วงสั้นๆ เพราะอาจเป็นปัญหาชั่วคราว (เน็ตหลุด / rate limit)
# ถ้าแคชถาวร ผู้ใช้ทุกคนหลังจากนั้นจะเห็นคำเตือนไปจนกว่าจะรีสตาร์ทแอป
SEMANTIC_FAIL_TTL = 60   # วินาที
_semantic_status = {"ok": False, "detail": "", "checked_at": None}

def check_semantic_search() -> tuple[bool, str]:
    """ทดสอบว่า semantic search ใช้งานได้จริง ไม่ใช่แค่เช็คว่ามีไฟล์ embeddings.npy อยู่หรือไม่
    เพราะ embeddings.npy อาจมีอยู่และตรงกับจำนวน chunk แต่ query-time embedding ยังพังได้
    (เช่น GEMINI_API_KEY หายไปจาก .env) โดยที่ embeddings_ready() มองไม่เห็นเคสนี้เลย"""
    if not client:   # ไม่มี key เป็นปัญหาถาวร ไม่ต้องลองใหม่
        return False, "ไม่มี GEMINI_API_KEY หรือ key ไม่ถูกต้อง (genai.Client สร้างไม่สำเร็จ)"
    checked_at = _semantic_status["checked_at"]
    if checked_at is not None and (
        _semantic_status["ok"] or time.monotonic() - checked_at < SEMANTIC_FAIL_TTL
    ):
        return _semantic_status["ok"], _semantic_status["detail"]
    try:
        vec = embed_query("ทดสอบระบบ")
        ok, detail = (False, "embed_query() คืนค่าว่าง") if vec is None or len(vec) == 0 else (True, "")
    except Exception as e:
        ok, detail = False, str(e)[:150]
    _semantic_status.update(ok=ok, detail=detail, checked_at=time.monotonic())
    return ok, detail

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])
    chunks, qa, emb = load_all_data()

    await cl.Message(
        content="สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ 🐍\n\n(ขับเคลื่อนด้วย Chainlit + DeepSeek / Gemini)"
    ).send()

    if emb is None:
        await cl.Message(content="ℹ️ คำเตือน: ไม่มีไฟล์ embeddings.npy ระบบจะใช้เฉพาะ Keyword Search").send()
    elif not embeddings_ready(emb, chunks):
        await cl.Message(content="⚠️ คำเตือน: embeddings.npy ไม่ตรงกับเนื้อหา ระบบจะปิด Semantic Search ชั่วคราว").send()
    else:
        ok, detail = await asyncio.to_thread(check_semantic_search)
        if not ok:
            await cl.Message(
                content=(
                    "⚠️ คำเตือน: ทดสอบ Semantic Search ไม่ผ่าน แม้ embeddings.npy จะมีอยู่ครบ "
                    f"(สาเหตุ: {detail}) ระบบจะยังลองใช้ Semantic Search ทุกคำถาม "
                    "และถ้าล้มเหลวจะค้นด้วย Keyword Search แทน "
                    "ซึ่งอาจตอบไม่ครบถ้วนในคำถามที่ใช้คำภาษาอังกฤษ/ไทยสลับกับเนื้อหาต้นฉบับ "
                    "— ตรวจสอบค่า GEMINI_API_KEY ใน .env หรือ environment secret"
                )
            ).send()

# ลำดับการตอบ: 1) OpenRouter แบบเสียเงิน  2) Gemini ฟรีต่อตรง  3) OpenRouter ฟรี
# DeepSeek V4.1 Flash: เทส 36 ครั้ง ตอบได้ 100% ตัวอักษรแรกขึ้นค่ากลาง 3.7 วิ ~$0.0006 ต่อคำถาม
# (Gemini ฟรีชุดเดียวกัน: 10.7 วิ และ quota หมดหลัง 5 ข้อ)
OPENROUTER_PAID_MODELS = ["deepseek/deepseek-v4.1-flash"]
# 3.6-flash ตอบครบ/นิ่งกว่า flash-lite — ถ้า quota หมดค่อยตกไป flash-lite
GEMINI_MODELS     = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
# ใช้เฉพาะโมเดลฟรี (:free) — ไม่เสียเครดิต แต่มักโดน rate limit จากต้นทางบ่อย จึงใส่ไว้หลายตัวเผื่อสลับ
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
    "cohere/north-mini-code:free",
]

STREAM_IDLE_TIMEOUT = 30   # วินาที — สตรีมเงียบนานกว่านี้ถือว่าค้าง แล้วข้ามไปโมเดลถัดไป

async def _with_idle_timeout(stream, seconds: float = STREAM_IDLE_TIMEOUT):
    """วน async stream โดยโยน TimeoutError ถ้าไม่มี chunk ใหม่มาภายใน seconds"""
    it = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(it.__anext__(), seconds)
        except StopAsyncIteration:
            return
        yield chunk

async def _reset(msg: cl.Message):
    """ล้างข้อความที่สตรีมค้างไว้จากโมเดลที่ล้มกลางทาง ก่อนลองโมเดลถัดไป"""
    if msg.content:
        msg.content = ""
        await msg.update()

async def answer_with_gemini(msg: cl.Message, raw_history: list, user_input: str) -> str | bool:
    if not client:
        return False
    gemini_history = [types.Content(role=h["role"], parts=[types.Part(text=h["content"])]) for h in raw_history]
    for model in GEMINI_MODELS:
        if _cooling(model):
            log.info(f"Gemini skip {model}: quota cooldown")
            continue
        try:
            await _reset(msg)
            # ใช้ client.aio (async) — ตัว sync จะบล็อก event loop ทั้งเซิร์ฟเวอร์ระหว่างรอ Gemini
            chat = client.aio.chats.create(model=model, config=CHAT_CONFIG, history=gemini_history)
            stream = await asyncio.wait_for(chat.send_message_stream(user_input), STREAM_IDLE_TIMEOUT)
            async for chunk in _with_idle_timeout(stream):
                if chunk.text:
                    await msg.stream_token(chunk.text)
            if not msg.content.strip():
                raise RuntimeError("empty answer")   # บางครั้งตอบสำเร็จแต่ไม่มีข้อความ
            await msg.update()
            log.info(f"Answered by {model}")
            return model
        except Exception as e:   # 404 / 429 / 503 / timeout -> ข้ามไปโมเดลถัดไปทันที ไม่ลองซ้ำ
            _note_failure("Gemini Fallback", model, e)
    return False

async def answer_with_openrouter(msg: cl.Message, raw_history: list, user_input: str,
                                 openrouter_key: str | None, models: list[str] | None = None) -> str | bool:
    if not openrouter_key:
        return False
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key,
                                timeout=60, max_retries=0)

    # แปลง role: "model" -> "assistant"
    or_history = [{"role": "system", "content": PROMPT_PYBOT}]
    for h in raw_history:
        role = "assistant" if h["role"] == "model" else h["role"]
        or_history.append({"role": role, "content": h["content"]})
    or_history.append({"role": "user", "content": user_input})

    for model_name in (OPENROUTER_MODELS if models is None else models):
        if _cooling(model_name):
            log.info(f"OpenRouter skip {model_name}: quota cooldown")
            continue
        try:
            await _reset(msg)
            stream = await openai_client.chat.completions.create(
                model=model_name,
                messages=or_history,
                temperature=0.0,
                max_tokens=2048,
                stream=True
            )
            async for chunk in _with_idle_timeout(stream):
                if chunk.choices and chunk.choices[0].delta.content:
                    await msg.stream_token(chunk.choices[0].delta.content)
            if not msg.content.strip():
                # โมเดลฟรีบางตัวตอบ HTTP 200 แต่ข้างในเป็น error ("Upstream error ... overloaded")
                raise RuntimeError("empty answer")
            await msg.update()
            log.info(f"Answered by {model_name}")
            return model_name
        except Exception as e:
            _note_failure("OpenRouter Fallback", model_name, e)
    return False

def build_raw_history(session_messages: list, relevant_context: str) -> list:
    """ประวัติแชทล่าสุด + เนื้อหาจากหนังสือที่ค้นมา ในรูป dict กลาง (ใช้ร่วมกับ streamlit_app.py)"""
    raw_history = [{"role": m["role"], "content": m["content"]}
                   for m in session_messages[-MAX_HISTORY_MESSAGES:]]
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
    return raw_history

async def answer(msg: cl.Message, raw_history: list, user_input: str, openrouter_key: str | None) -> str | bool:
    """ลองตอบตามลำดับ: OpenRouter เสียเงิน -> Gemini ฟรี -> OpenRouter ฟรี (ใช้ร่วมกับ streamlit_app.py)
    คืนชื่อโมเดลที่ตอบได้ หรือ False ถ้าล้มทุกตัว"""
    return (await answer_with_openrouter(msg, raw_history, user_input, openrouter_key, OPENROUTER_PAID_MODELS)
            or await answer_with_gemini(msg, raw_history, user_input)
            or await answer_with_openrouter(msg, raw_history, user_input, openrouter_key))

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
        # ฟังก์ชันพวกนี้เรียก API แบบ sync — รันใน thread เพื่อไม่ให้บล็อกผู้ใช้คนอื่น
        expanded = await asyncio.to_thread(rewrite_query, user_input, chunks)
        relevant_context = await asyncio.to_thread(
            search, expanded, chunks, qa_rows, embeddings=embeddings, embed_fn=embed_query,
            semantic_query=user_input)
        step.output = relevant_context
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    session_messages = cl.user_session.get("messages")
    raw_history = build_raw_history(session_messages, relevant_context)
    
    # เตรียมส่งข้อความแบบสตรีม
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        if not client and not openrouter_key:
            await cl.Message(content="❌ ไม่พบ Gemini API Key หรือ OpenRouter API Key").send()
            return
        success = await answer(msg, raw_history, user_input, openrouter_key)

        if not success:
            await cl.Message(content="⚠️ ไม่สามารถเชื่อมต่อกับโมเดลใดๆ ได้ในขณะนี้ กรุณาตรวจสอบ API Key หรือลองใหม่ภายหลัง").send()
        else:
            # บันทึกประวัติ
            session_messages.append({"role": "user", "content": user_input})
            session_messages.append({"role": "model", "content": msg.content})
            cl.user_session.set("messages", session_messages)

    except Exception as e:
        await cl.Message(content=f"❌ เกิดข้อผิดพลาดจาก API: {str(e)}").send()
