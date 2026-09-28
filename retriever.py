import re
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path

try:                                    # ตัดคำไทยระดับคำ (ช่วย keyword match มาก)
    from pythainlp.tokenize import word_tokenize as _th_tokenize
except Exception:                       # ไม่มี pythainlp -> fallback เป็น regex
    _th_tokenize = None

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[฀-๿]+")


@lru_cache(maxsize=8192)
def _tokens(text: str) -> frozenset:
    """แตกข้อความเป็นเซตของคำ (ไทยตัดด้วย pythainlp, อังกฤษ/เลขด้วย regex)"""
    out: set[str] = set()
    for m in _TOKEN_RE.findall(text.lower()):
        if m[0].isascii():
            out.add(m)
        elif _th_tokenize is not None:
            out.update(w.strip() for w in _th_tokenize(m, keep_whitespace=False) if w.strip())
        else:
            out.add(m)
    return frozenset(out)

# ---------- chunking ----------
TARGET_CHARS   = 1000     # เป้าหมายขนาดต่อ chunk (ตัวอักษร)
MAX_CHARS      = 1600     # เกินนี้บังคับหั่น
OVERLAP_CHARS  = 150      # overlap ระหว่าง sub-chunk ของ section เดียวกัน

# ---------- retrieval ----------
PDF_TOP_K   = 8           # ดึง chunk จากหนังสือเยอะขึ้นเพื่อให้ครอบคลุมเนื้อหา
EXCEL_TOP_K = 3
SEM_POOL    = 20          # จำนวน candidate จาก semantic ก่อน fuse (เพิ่มเพื่อจับเนื้อหาที่เกี่ยวข้องแม้ไม่ตรง)
KW_POOL     = 20          # จำนวน candidate จาก keyword ก่อน fuse
MIN_SIM     = 0.25        # ลดลงเพื่อให้จับ semantic match ที่ "เกี่ยวข้อง" แม้ไม่ตรงทั้งหมด
RRF_K       = 60          # ค่าคงที่ Reciprocal Rank Fusion
DEFAULT_PER_HEAD_CAP  = 3 # จำกัด chunk/หัวข้อ ปกติ เพื่อให้ context ครอบคลุมหลายหัวข้อ ไม่กระจุก
DOMINANT_PER_HEAD_CAP = 6 # ยกเว้นหัวข้อที่ตรงกับคำถามที่สุด (rank 1) ให้ดึงได้มากกว่า เพราะคำถาม
                          # แบบ "มีกี่ข้อ/กี่ตัว" มักต้องใช้ทุก sub-chunk ของหัวข้อเดียวกันจึงจะครบ

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*#*$')
# บางหัวข้อในต้นฉบับเป็นตัวหนาล้วน ไม่มี # นำหน้า เช่น "**3. ประวัติการสร้างภาษาไพธอน...**"
# (พบ ~40 จุดทั่วเล่ม จากการแปลง PDF ที่ไม่สม่ำเสมอ) — ถ้าไม่จับเป็นหัวข้อ เนื้อหาทั้งหมด
# ใต้บรรทัดนั้นจะไหลเข้าไปติด breadcrumb ของหัวข้อก่อนหน้าไปเรื่อยๆ จนกว่าจะเจอ ### จริงถัดไป
BOLD_HEADING_RE = re.compile(r'^\*\*(\d+(?:\.\d+)*\.?)\s+(.+?)\*\*\s*$')
BOLD_HEADING_LEVEL = 3   # เทียบเท่าหัวข้อย่อยระดับ ### ของหัวข้อจริงแบบเดียวกันในเล่มนี้
# แต่บางหัวข้อตัวหนาเป็นหัวข้อ "ลูก" ที่เริ่มนับ 1 ใหม่ใต้ ### (เช่น "### 6. ...GUI" ตามด้วย
# "**1. เครื่องคิดเลข...**") — ถ้าให้ระดับเท่ากันจะเบียดหัวข้อแม่ออกจาก breadcrumb
LEADING_NUM_RE = re.compile(r'^(\d+)')
# หัวข้อขยะจากการแปลง PDF (เลขหน้า / "จบภาค X" / "ภ X") — ไม่เอาเข้า breadcrumb
JUNK_HEADING_RE = re.compile(r'^(จบภาค|ภาค|ภ|บท|หน้า)?\s*[\d.\)]*\s*$')
CRUMB_DEPTH = 2          # จำนวนชั้นหัวข้อสูงสุดใน breadcrumb

# คำถามทั่วไปที่ไม่ช่วย discriminate — ตัดออกจาก score
THAI_STOPWORDS = {
    "เขียน", "ยังไง", "คืออะไร", "ใช้ยังไง", "อะไร", "ทำไม", "อย่างไร",
    "หน่อย", "ด้วย", "ครับ", "ค่ะ", "นะ", "บ้าง", "ได้", "ให้", "แบบ",
    "ตัวอย่าง", "วิธี", "การ", "ของ", "ที่", "เป็น", "มี", "ใน",
    "กับ", "และ", "หรือ", "แต่", "จะ", "ก็", "แล้ว", "จาก", "โดย",
}

PY_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "return",
    "import", "from", "try", "except", "with", "as", "and", "or", "not",
    "in", "is", "lambda", "pass", "break", "continue", "yield", "global",
    "print", "list", "dict", "tuple", "set", "range", "len", "type",
    "str", "int", "float", "bool", "none", "true", "false", "input",
    "open", "append", "remove", "sort", "map", "filter", "zip",
}


def clean_thai(text: str) -> str:
    """แก้ encoding artifact จาก PDF"""
    # ำ ที่ถูก decode เป็น space + สระอา
    text = re.sub(r'([ก-ฮ]) า', r'\1ำ', text)
    # สระ/วรรณยุกต์ลอย ที่หลุด space นำหน้า
    text = re.sub(r'([ก-ฮ]) ([ัิ-ู็-๎])', r'\1\2', text)
    # space แปลก ๆ หน้า ำ / ะ ที่ยังเหลือ (เช่น "ค ำน ำ" -> "คำนำ")
    text = re.sub(r'([ก-ฮ]) ([ำะ])', r'\1\2', text)
    return text


# ---------------------------------------------------------------------------
#  Markdown-aware chunking
# ---------------------------------------------------------------------------
def _blocks(body: str) -> list[str]:
    """แตก body เป็น block ตามบรรทัดว่าง โดยไม่แตกกลาง code fence"""
    blocks, cur, in_fence = [], [], False
    for line in body.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not s and not in_fence:
            if cur:
                blocks.append('\n'.join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append('\n'.join(cur).strip())
    return [b for b in blocks if b]


def _pack(body: str) -> list[str]:
    """รวม block ให้ได้ chunk ขนาด ~TARGET_CHARS พร้อม overlap เล็กน้อย"""
    if len(body) <= MAX_CHARS:
        return [body]
    pieces, cur = [], ""
    for b in _blocks(body):
        if len(b) > MAX_CHARS:                       # block เดียวยาวเกิน -> หั่นตามบรรทัด
            if cur.strip():
                pieces.append(cur.strip())
            cur = ""
            sub = ""
            for ln in b.split('\n'):
                if sub and len(sub) + len(ln) > MAX_CHARS:
                    pieces.append(sub.strip())
                    sub = ""
                sub += ln + "\n"
            cur = sub
            continue
        if cur and len(cur) + len(b) + 2 > TARGET_CHARS:
            tail = cur[-OVERLAP_CHARS:]
            sp = tail.find(' ')
            if sp != -1:
                tail = tail[sp + 1:]
            pieces.append(cur.strip())
            cur = tail + "\n\n" + b
        else:
            cur = (cur + "\n\n" + b) if cur else b
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def _bold_level(num: int, stack: list[tuple[int, str, int | None]]) -> int:
    """เดาระดับของหัวข้อตัวหนาจากความต่อเนื่องของเลขข้อ:
    ต่อจากหัวข้อลูกก่อนหน้า -> ลูก, ต่อจาก ### -> พี่น้องของ ###, เริ่ม 1 ใหม่ใต้ ### -> ลูก"""
    last = {lvl: n for lvl, _, n in stack}
    child = BOLD_HEADING_LEVEL + 1
    if last.get(child) == num - 1:
        return child
    if last.get(BOLD_HEADING_LEVEL) == num - 1:
        return BOLD_HEADING_LEVEL
    if num == 1 and BOLD_HEADING_LEVEL in last:
        return child
    return BOLD_HEADING_LEVEL


def load_chunks(md_path: str) -> list[str]:
    """หั่นหนังสือตามลำดับชั้นหัวข้อ Markdown แล้วแนบ breadcrumb (h1 > h2 > h3)
    ไว้บรรทัดแรกของทุก chunk เพื่อให้ทั้ง keyword และ semantic search เห็นบริบทหัวข้อ"""
    text = clean_thai(Path(md_path).read_text(encoding="utf-8"))
    stack: list[tuple[int, str, int | None]] = []   # [(level, title, เลขนำหน้า), ...]
    buf: list[str] = []
    chunks: list[str] = []
    in_fence = False

    def flush():
        body = '\n'.join(buf).strip()
        buf.clear()
        if not body:
            return
        crumb = ' > '.join(t for _, t, _ in stack[-CRUMB_DEPTH:])
        for piece in _pack(body):
            chunks.append(f"{crumb}\n\n{piece}" if crumb else piece)

    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = HEADING_RE.match(s) if not in_fence else None
        bm = BOLD_HEADING_RE.match(s) if (not in_fence and not m) else None
        if m or bm:
            flush()
            if m:
                level = len(m.group(1))
                title = re.sub(r'[*_`]+', '', m.group(2)).strip()
                num_m = LEADING_NUM_RE.match(title)
                num = int(num_m.group(1)) if num_m else None
            else:
                num = int(LEADING_NUM_RE.match(bm.group(1)).group(1))
                level = _bold_level(num, stack)
                title = bm.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            if len(title) >= 3 and not JUNK_HEADING_RE.match(title):
                stack.append((level, title, num))
                buf.append(line)               # เก็บบรรทัดหัวข้อจริงไว้ในเนื้อ chunk ด้วย
            # หัวข้อขยะ (เลขหน้า / "จบภาค X") ทิ้งไปเลย ไม่ให้ปนเนื้อหา
        else:
            buf.append(line)
    flush()
    return chunks


def load_embeddings(path: str) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None


def load_qa(xlsx_path: str) -> list[dict]:
    df = pd.read_excel(xlsx_path).fillna("")
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "topic": f"{row['หัวข้อหลัก']} {row['หัวข้อย่อย']} {row['ย่อยรอง']}",
            "question": str(row["คำถาม (แก้ไขใหม่)"]),
            "answer": str(row["คำตอบ"]),
            "keywords": str(row["keyword ตรง"]),
        })
    return rows


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------
def _score(query: str, text: str) -> float:
    q_words = _tokens(query) - THAI_STOPWORDS
    if not q_words:
        return 0.0
    tl = text.lower()
    t_words = _tokens(text)
    shared = q_words & t_words
    score = float(len(shared))

    # นับความถี่ของคำอังกฤษ/keyword ที่ตรงกัน (ช่วยแยก chunk ที่ "พูดถึงเรื่องนี้จริง ๆ"
    # ออกจาก chunk ที่แค่บังเอิญอยู่ใต้หัวข้อเดียวกัน)
    for w in shared:
        if w.isascii() and len(w) >= 2:
            score += min(len(re.findall(rf'\b{re.escape(w)}\b', tl)), 5) * 0.6

    # boost ถ้าคำใน query ตรงกับ breadcrumb/heading (บรรทัดแรกของ chunk)
    if '\n' in text:
        h_words = _tokens(text.split('\n', 1)[0])
        score += len(q_words & h_words) * 4

    # boost Python keyword ที่ตรงกัน
    score += len((q_words & PY_KEYWORDS) & (t_words & PY_KEYWORDS)) * 3
    return score


def search_qa(query: str, qa_rows: list[dict], top_k: int = EXCEL_TOP_K) -> str:
    clean_q = clean_thai(query)
    scores = []
    for i, row in enumerate(qa_rows):
        score = (
            _score(clean_q, row["question"]) * 3
            + _score(clean_q, row["keywords"]) * 2
            + _score(clean_q, row["topic"])
        )
        scores.append((score, i))
    scores.sort(reverse=True)
    best = [(s, i) for s, i in scores[:top_k] if s > 0]
    if not best:
        return ""
    parts = [f"Q: {qa_rows[i]['question']}\nA: {qa_rows[i]['answer']}" for _, i in best]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
#  Hybrid PDF search  (semantic + keyword, fused with RRF)
# ---------------------------------------------------------------------------
def _semantic_rank(query_vec: np.ndarray, embeddings: np.ndarray,
                   pool: int) -> list[tuple[int, float]]:
    denom = (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)) + 1e-9
    sims = np.dot(embeddings, query_vec) / denom
    order = np.argsort(sims)[::-1][:pool]
    return [(int(i), float(sims[i])) for i in order]


def _rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    agg: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg, key=agg.get, reverse=True)


def embeddings_ready(embeddings: np.ndarray | None, chunks: list[str]) -> bool:
    """True เมื่อ embeddings ใช้ค้น semantic ได้จริง (จำนวนแถวตรงกับจำนวน chunk ปัจจุบัน)
    ถ้าไม่ตรง (เช่นแก้วิธีหั่น chunk แล้วลืมรัน build_embeddings.py ใหม่) search_pdf()
    จะปิด semantic ให้เองเงียบ ๆ — ใช้ฟังก์ชันนี้เพื่อแจ้งเตือนผู้ใช้แทนความเงียบนั้น"""
    return embeddings is not None and len(embeddings) == len(chunks)


def _kw_rank(query: str, chunks: list[str]) -> tuple[list[int], list[tuple[float, int]]]:
    scored = sorted(((_score(query, c), i) for i, c in enumerate(chunks)), reverse=True)
    return [i for s, i in scored[:KW_POOL] if s > 0], scored


def search_pdf(query: str, chunks: list[str],
               top_k: int = PDF_TOP_K,
               embeddings: np.ndarray | None = None,
               embed_fn=None,
               semantic_query: str | None = None) -> str:
    """query = คำถามที่ขยาย keyword แล้ว (ช่วย keyword match คำสะกดผิด/ไทย-อังกฤษ)
    semantic_query = คำถามเดิมของผู้ใช้ — ใช้ทำ semantic search เพราะ keyword ที่ LLM เติมมา
    (เช่น "Built-in functions") ดึง embedding ออกนอกเรื่องได้ และใช้ทำ keyword search อีกชุดด้วย"""
    clean_q = clean_thai(query)
    clean_sem_q = clean_thai(semantic_query) if semantic_query else clean_q

    # --- keyword ranking (ใช้ทุกโหมด) ---
    kw_rank, kw_scored = _kw_rank(clean_q, chunks)
    kw_rank_orig = _kw_rank(clean_sem_q, chunks)[0] if semantic_query else []

    # --- semantic ranking (ถ้ามี embeddings ที่ตรงกับ chunks) ---
    sem_rank: list[int] = []
    if embeddings is not None and embed_fn is not None and embeddings_ready(embeddings, chunks):
        try:
            ranked = _semantic_rank(embed_fn(clean_sem_q), embeddings, SEM_POOL)
            sem_rank = [i for i, sim in ranked if sim >= MIN_SIM] or [ranked[0][0]]
        except Exception:
            sem_rank = []                      # fall back to keyword only

    rankings = [r for r in (sem_rank, kw_rank_orig, kw_rank) if r]
    ordered = _rrf(*rankings) if rankings else [i for _, i in kw_scored[:2]]

    # เลือกไม่เกิน DEFAULT_PER_HEAD_CAP chunk ต่อหัวข้อ เพื่อให้ครอบคลุมหลายหัวข้อไม่กระจุกเกินไป
    # ยกเว้นหัวข้อที่ rank อันดับ 1 (ตรงกับคำถามที่สุด) ให้ดึงได้ถึง DOMINANT_PER_HEAD_CAP
    # เพราะคำถามที่คำตอบสมบูรณ์ต้องมาจากหัวข้อเดียวกันทั้งหมด (เช่น "มีกี่ข้อ/กี่ตัว") ไม่ควรถูก
    # หัวข้ออื่นที่ไม่เกี่ยวข้องแย่ง slot ไปจนคำตอบขาดหาย
    dominant_head = chunks[ordered[0]].split('\n', 1)[0] if ordered else None
    per_head: dict[str, int] = {}
    picked: list[int] = []
    for i in ordered:
        h = chunks[i].split('\n', 1)[0]
        cap = DOMINANT_PER_HEAD_CAP if h == dominant_head else DEFAULT_PER_HEAD_CAP
        if per_head.get(h, 0) >= cap:
            continue
        per_head[h] = per_head.get(h, 0) + 1
        picked.append(i)
        if len(picked) >= top_k:
            break
    if not picked:
        # fallback: ถ้าไม่เจอเลย ให้เอา chunk ที่ score สูงสุด 2 ตัวแรกมาแทน
        # เพื่อให้โมเดลยังมีเนื้อหาพิจารณา แทนที่จะส่ง context ว่าง
        picked = ordered[:min(2, top_k)] if ordered else [i for _, i in kw_scored[:2]]

    return "\n\n---\n\n".join(chunks[i] for i in picked)


def search(query: str, chunks: list[str], qa_rows: list[dict],
           embeddings: np.ndarray | None = None,
           embed_fn=None,
           semantic_query: str | None = None) -> str:
    qa_result = search_qa(query, qa_rows)
    pdf_result = search_pdf(query, chunks,
                            embeddings=embeddings, embed_fn=embed_fn,
                            semantic_query=semantic_query)
    parts = []
    # เนื้อหาจากหนังสือเป็นแหล่งข้อมูลหลัก — วางก่อนเสมอ
    if pdf_result:
        parts.append(f"[จากหนังสือ Markdown — แหล่งข้อมูลหลัก]\n{pdf_result}")
    if qa_result:
        parts.append(f"[จาก Q&A Excel — ข้อมูลเสริม]\n{qa_result}")
    return "\n\n===\n\n".join(parts)

