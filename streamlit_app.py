"""PyBot เวอร์ชัน Streamlit (สำหรับ Streamlit Community Cloud)

ใช้ logic ชุดเดียวกับ cl_app.py ทั้งหมด (ค้นหนังสือ / ลำดับโมเดล / fallback) — แก้ที่ cl_app.py ที่เดียว
รันในเครื่อง:  streamlit run streamlit_app.py
บน Streamlit Cloud: ใส่ GEMINI_API_KEY / OPENROUTER_API_KEY ใน Secrets (ระดับบนสุด จะกลายเป็น env var ให้เอง)
"""
import asyncio
import os

import streamlit as st

import cl_app as core
from retriever import search

GREETING_REPLY = "สวัสดีครับ! ผม PyBot ผู้ช่วยเรียน Python จากหนังสือ Python MSU ครับ ถามเรื่อง Python ได้เลยครับ 🐍"

st.set_page_config(page_title="PyBot — Python MSU", page_icon="🐍")
st.title("🐍 PyBot")
st.caption("ผู้ช่วยเรียน Python จากหนังสือ Python MSU (ขับเคลื่อนด้วย DeepSeek / Gemini)")


@st.cache_resource(show_spinner="กำลังโหลดเนื้อหาหนังสือ...")
def load_data():
    return core.load_all_data()


@st.cache_resource(ttl=core.SEMANTIC_FAIL_TTL)
def semantic_status():
    return core.check_semantic_search()


def run_async(coro):
    """แต่ละ session ของ Streamlit รันใน thread ของตัวเอง — ให้แต่ละ session มี event loop ถาวรของตัวเอง"""
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
    return st.session_state.loop.run_until_complete(coro)


class StreamlitMessage:
    """ตัวแปลงให้ answer_with_gemini / answer_with_openrouter สตรีมลง Streamlit แทน cl.Message"""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.content = ""

    async def stream_token(self, token: str):
        self.content += token
        self.placeholder.markdown(self.content + "▌")

    async def update(self):
        self.placeholder.markdown(self.content)


chunks, qa_rows, embeddings = load_data()

if embeddings is None:
    st.info("ℹ️ ไม่มีไฟล์ embeddings.npy ระบบจะใช้เฉพาะ Keyword Search")
elif not core.embeddings_ready(embeddings, chunks):
    st.warning("⚠️ embeddings.npy ไม่ตรงกับเนื้อหา ระบบจะปิด Semantic Search ชั่วคราว")
else:
    ok, detail = semantic_status()
    if not ok:
        st.warning(
            f"⚠️ ทดสอบ Semantic Search ไม่ผ่าน (สาเหตุ: {detail}) ระบบจะยังลองใช้ Semantic Search ทุกคำถาม "
            "และถ้าล้มเหลวจะค้นด้วย Keyword Search แทน — ตรวจสอบค่า GEMINI_API_KEY ใน Secrets"
        )

# ประวัติแชท: role "user" / "model" แบบเดียวกับ cl_app.py
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

user_input = st.chat_input("ถามเรื่อง Python ได้เลย...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if core.is_greeting(user_input):
            st.markdown(GREETING_REPLY)
            st.stop()

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not core.client and not openrouter_key:
            st.error("❌ ไม่พบ Gemini API Key หรือ OpenRouter API Key")
            st.stop()

        with st.status("🔍 กำลังค้นหาข้อมูลในหนังสือ...", expanded=False) as status:
            expanded = core.rewrite_query(user_input, chunks)
            relevant_context = search(expanded, chunks, qa_rows, embeddings=embeddings,
                                      embed_fn=core.embed_query, semantic_query=user_input)
            st.text(relevant_context)
            status.update(label="🔍 ค้นหาข้อมูลในหนังสือเสร็จแล้ว", state="complete")

        raw_history = core.build_raw_history(st.session_state.messages, relevant_context)
        msg = StreamlitMessage(st.empty())

        async def answer() -> bool:
            # ลำดับเดียวกับ cl_app.py: OpenRouter เสียเงิน -> Gemini ฟรี -> OpenRouter ฟรี
            return await core.answer(msg, raw_history, user_input, openrouter_key)

        try:
            success = run_async(answer())
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดจาก API: {e}")
            st.stop()

        if not success:
            st.warning("⚠️ ไม่สามารถเชื่อมต่อกับโมเดลใดๆ ได้ในขณะนี้ กรุณาลองใหม่ภายหลัง")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "model", "content": msg.content})
            st.caption(f"ตอบโดย: {success}")
