---
title: PyBot Python MSU
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🐍 PyBot — ผู้ช่วยเรียน Python (Python MSU)

RAG chatbot ตอบคำถาม Python จากหนังสือ **"เชี่ยวชาญการเขียนโปรแกรมด้วยภาษาไพธอน"**
ของ ผศ.สุชาติ คุ้มมะณี (มหาวิทยาลัยมหาสารคาม) + ชุด Q&A 197 ข้อ

## แหล่งข้อมูล
- `python_data.md` — เนื้อหาหนังสือ (แปลงจาก PDF)
- `python_qa.xlsx` — คำถาม-คำตอบ 197 ข้อ

## Stack
- **Chainlit** — UI
- **Google Gemini API** — LLM + embeddings
- Retrieval: keyword scoring (`retriever.py`) + semantic search (ถ้ามี `embeddings.npy`)

## รันในเครื่อง
```bash
pip install -r requirements.txt
cp .env.example .env          # แล้วใส่ GEMINI_API_KEY ของจริง
chainlit run cl_app.py
```

## รันด้วย Docker
```bash
docker build -t pybot .
docker run -p 7860:7860 --env-file .env pybot
```

## Deploy

### Hugging Face Spaces (Docker SDK)
1. สร้าง Space ใหม่ → SDK = **Docker**
2. push repo นี้เข้า Space (มี `Dockerfile` อยู่แล้ว)
3. Space **Settings → Variables and secrets** → เพิ่ม secret `GEMINI_API_KEY`

### Render / Railway / Fly.io
- Deploy จาก `Dockerfile` ได้ตรงๆ (แพลตฟอร์มพวกนี้จะ inject ตัวแปร `$PORT` เองผ่าน `docker run -e PORT=...`)
- ตั้ง env var `GEMINI_API_KEY` ในหน้า dashboard ของแต่ละแพลตฟอร์ม

## เปิด semantic search (แนะนำ)
ถ้าไม่มี `embeddings.npy` แอปจะใช้ keyword search อย่างเดียว (retriever เป็น hybrid
RRF: semantic + keyword — จะทำงานเต็มรูปแบบเมื่อมีไฟล์นี้)
```bash
python build_embeddings.py     # ~2 นาที (batch), สร้าง embeddings.npy
git add embeddings.npy && git commit -m "add embeddings" && git push
```
> ถ้าแก้การหั่น chunk ใน `retriever.py` ต้องรัน `build_embeddings.py` ใหม่เสมอ

## หมายเหตุ
- ต้องมี GEMINI_API_KEY จาก https://aistudio.google.com/apikey
- โมเดลที่ใช้กำหนดใน `cl_app.py` (`gemini-3.5-flash-lite`, `gemini-3.6-flash`, ...) — หากบัญชีไม่มีโมเดลนี้ ให้เปลี่ยนเป็น `gemini-2.0-flash` / `gemini-2.5-flash`
