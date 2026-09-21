"""
รัน script นี้ครั้งเดียวเพื่อสร้าง embeddings.npy
  python build_embeddings.py
สามารถรันซ้ำได้ถ้าหยุดกลางทาง — จะต่อจากที่ค้างไว้ (checkpoint = embeddings_tmp.npy)
checkpoint จะถูกตรวจกับจำนวน chunk ปัจจุบันก่อนเสมอ ถ้าไม่ตรง (เช่นแก้วิธีหั่น chunk
ใน retriever.py แล้วมี checkpoint เก่าค้างอยู่) จะทิ้ง checkpoint เก่าแล้วเริ่มใหม่ทั้งหมด
แทนที่จะ resume ทับด้วยเวกเตอร์ที่ตรงกับ chunk คนละชุด

ซ่อมเฉพาะ chunk ที่ embed ไม่สำเร็จ (กลายเป็นเวกเตอร์ศูนย์ใน embeddings.npy ที่มีอยู่แล้ว):
  python build_embeddings.py --repair

หมายเหตุ: gemini-embedding-001 free tier จำกัด 100 requests/นาที
สคริปต์นี้จึงหน่วง ~0.7s/ครั้ง (~85/นาที) ใช้เวลาราว 18 นาทีสำหรับ ~1,500 chunks
"""
import os, sys, time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retriever import load_chunks

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR  = os.path.dirname(__file__)
MD_PATH   = os.path.join(BASE_DIR, "python_data.md")
OUT_PATH  = os.path.join(BASE_DIR, "embeddings.npy")
TMP_PATH  = os.path.join(BASE_DIR, "embeddings_tmp.npy")
META_PATH = os.path.join(BASE_DIR, "embeddings_tmp_meta.txt")  # จำนวน chunk ตอนสร้าง checkpoint
MODEL     = "gemini-embedding-001"
DIM       = 3072
DELAY     = 0.7          # หน่วงปกติระหว่าง request (วินาที) -> ~85/นาที < โควตา 100
CKPT      = 25           # save checkpoint ทุกกี่ chunk

CFG = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")

REPAIR = "--repair" in sys.argv

chunks = load_chunks(MD_PATH)
total  = len(chunks)
print(f"Chunks: {total}", flush=True)


def embed_one(text: str) -> list[float]:
    delay = 2.0
    for attempt in range(8):
        try:
            r = client.models.embed_content(model=MODEL, contents=text[:2000], config=CFG)
            return r.embeddings[0].values
        except Exception as e:
            is_quota = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            wait = 15.0 if is_quota else delay
            print(f"  retry {attempt+1} (wait {wait:.0f}s): {str(e)[:90]}", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 30)
    print("  SKIP -> zero vector", flush=True)
    return [0.0] * DIM


# ---------------------------------------------------------------------------
#  โหมดซ่อม — เติมเฉพาะแถวที่เป็นเวกเตอร์ศูนย์ใน embeddings.npy ที่มีอยู่แล้ว
# ---------------------------------------------------------------------------
if REPAIR:
    if not os.path.exists(OUT_PATH):
        print("ไม่พบ embeddings.npy — รันแบบปกติ (ไม่ใส่ --repair) ก่อนครั้งแรก", flush=True)
        sys.exit(1)
    arr = np.load(OUT_PATH)
    if arr.shape[0] != total:
        print(f"จำนวนแถวใน embeddings.npy ({arr.shape[0]}) ไม่ตรงกับจำนวน chunk ปัจจุบัน "
              f"({total}) — ต้องรันแบบเต็มใหม่ (ไม่ใส่ --repair) แทน", flush=True)
        sys.exit(1)

    zero_idx = np.where((arr == 0).all(axis=1))[0]
    print(f"พบเวกเตอร์ศูนย์ {len(zero_idx)}/{total} แถว — กำลังซ่อม", flush=True)
    for n, i in enumerate(zero_idx):
        arr[i] = embed_one(chunks[i])
        if (n + 1) % CKPT == 0 or n == len(zero_idx) - 1:
            np.save(OUT_PATH, arr)
            print(f"  checkpoint {n + 1}/{len(zero_idx)}", flush=True)
        time.sleep(DELAY)

    still_zero = int((arr == 0).all(axis=1).sum())
    print(f"Done -> embeddings.npy ซ่อมแล้ว  shape={arr.shape}  zero-vectors เหลือ={still_zero}", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
#  โหมดปกติ — สร้างใหม่ทั้งหมด (หรือ resume จาก checkpoint ที่ตรวจแล้วว่าใช้ได้)
# ---------------------------------------------------------------------------
if os.path.exists(TMP_PATH) and os.path.exists(META_PATH):
    saved_total = int(open(META_PATH, encoding="utf-8").read().strip() or -1)
    if saved_total != total:
        print(f"checkpoint เก่า ({saved_total} chunk) ไม่ตรงกับจำนวน chunk ปัจจุบัน ({total}) "
              f"— เนื้อหาหรือวิธีหั่น chunk เปลี่ยนไปแล้ว ทิ้ง checkpoint เก่าแล้วเริ่มใหม่ทั้งหมด", flush=True)
        os.remove(TMP_PATH)
        os.remove(META_PATH)
        all_vecs, start = [], 0
    else:
        all_vecs = list(np.load(TMP_PATH))
        start = len(all_vecs)
        print(f"Resume from {start}/{total}", flush=True)
elif os.path.exists(TMP_PATH):
    # checkpoint เก่าไม่มี meta กำกับ (มาจากสคริปต์เวอร์ชันก่อนหน้า) -> ไม่รู้ว่าตรงกับ
    # chunk ชุดปัจจุบันหรือไม่ ปลอดภัยกว่าจะเริ่มใหม่ทั้งหมดแทนการเดา
    print("พบ embeddings_tmp.npy เก่าที่ไม่มีข้อมูลกำกับ (meta) -> เริ่มใหม่ทั้งหมดเพื่อความปลอดภัย", flush=True)
    os.remove(TMP_PATH)
    all_vecs, start = [], 0
else:
    all_vecs, start = [], 0

with open(META_PATH, "w", encoding="utf-8") as f:
    f.write(str(total))

for i in range(start, total):
    all_vecs.append(embed_one(chunks[i]))
    if (i + 1) % CKPT == 0 or i == total - 1:
        np.save(TMP_PATH, np.array(all_vecs, dtype=np.float32))
        print(f"  checkpoint {i + 1}/{total}", flush=True)
    time.sleep(DELAY)

arr = np.array(all_vecs, dtype=np.float32)
np.save(OUT_PATH, arr)
if os.path.exists(TMP_PATH):
    os.remove(TMP_PATH)
if os.path.exists(META_PATH):
    os.remove(META_PATH)
zeros = int((arr == 0).all(axis=1).sum())
print(f"Done -> embeddings.npy  shape={arr.shape}  zero-vectors={zeros}", flush=True)
