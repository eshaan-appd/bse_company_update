import os, io, re, time, json, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
import streamlit as st

# IR / NLP
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# PDF stack (pure pip — OK on Streamlit Cloud)
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# =========================
# Config & constants
# =========================
st.set_page_config(page_title="BSE Company Update – AI Agent", layout="wide")
IST = pytz.timezone("Asia/Kolkata")

DATA_DIR = Path("storage")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_PATH = DATA_DIR / "bse_company_update.parquet"
CONFIG_PATH = DATA_DIR / "config.json"

BSE_BASE_PAGE = "https://www.bseindia.com/corporates/ann.html"
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_BASES = [
    "https://www.bseindia.com/xml-data/corpfiling/AttachHis/",
    "https://www.bseindia.com/xml-data/corpfiling/Attach/",
    "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
]
ILLEGAL_XML_RX = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')

def _clean(s: str) -> str:
    return ILLEGAL_XML_RX.sub("", s) if isinstance(s, str) else s

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _now_ist():
    return dt.datetime.now(IST)

# =========================
# Fetch announcements (PDF-free)
# =========================
def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   request_timeout: int = 25,
                                   verbose: bool = False) -> pd.DataFrame:
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": BSE_BASE_PAGE,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    try:
        s.get(BSE_BASE_PAGE, timeout=15)
    except Exception:
        pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1,
            "strCat": "-1",
            "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd,
            "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"],
            "strscrip": "",
            "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(BSE_API, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type", "")
            if "application/json" not in ct:
                if verbose: st.info(f"Non-JSON on page {page} for variant {v}.")
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None
            if not table:
                break
            params["pageno"] += 1
            page += 1
            time.sleep(0.25)
            if total and len(rows) >= total:
                break
        if rows:
            all_rows = rows
            break

    if not all_rows:
        return pd.DataFrame()

    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    return pd.DataFrame(all_rows, columns=list(all_keys))

# =========================
# Filter: Category + (optional) ticker/company
# =========================
def filter_by_category(df: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
    if df.empty:
        return df
    cat_col = _first_col(df, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
    if not cat_col:
        return df  # cannot filter without category column
    out = df.copy()
    out["_cat_norm"] = out[cat_col].map(_norm)
    out = out.loc[out["_cat_norm"] == _norm(category_filter)].drop(columns=["_cat_norm"])
    return out

def apply_company_filters(df: pd.DataFrame,
                          scrip_codes: str = "",
                          company_keywords: str = "",
                          exact_company: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    if scrip_codes.strip():
        codes = {c.strip() for c in re.split(r"[,\s]+", scrip_codes.strip()) if c.strip()}
        sc_col = _first_col(out, ["SCRIP_CD","SCRIPCODE","BSE_CODE"])
        if sc_col:
            out = out[out[sc_col].astype(str).isin(codes)]

    if company_keywords.strip():
        keys = [k.strip() for k in company_keywords.split(",") if k.strip()]
        name_col = _first_col(out, ["SLONGNAME","SNAME","COMPANY","COMPANYNAME"])
        if name_col:
            if exact_company:
                targets = {_norm(k) for k in keys}
                out = out[out[name_col].map(lambda x: _norm(x) in targets)]
            else:
                pat = "|".join([re.escape(k) for k in keys])
                out = out[out[name_col].str.contains(pat, case=False, na=False, regex=True)]

    return out

# =========================
# PDF text + tables extraction (no OCR in Cloud)
# =========================
def _text_pymupdf(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = [page.get_text("text") or "" for page in doc]
        return "\n".join(parts).strip()
    except Exception:
        return ""

def _text_pdfminer(pdf_bytes: bytes) -> str:
    try:
        return (pdfminer_extract_text(io.BytesIO(pdf_bytes)) or "").strip()
    except Exception:
        return ""

def _text_pypdf(pdf_bytes: bytes) -> str:
    try:
        rdr = PdfReader(io.BytesIO(pdf_bytes))
        out = []
        for p in rdr.pages:
            try:
                t = p.extract_text() or ""
                if t: out.append(t)
            except Exception:
                pass
        return "\n".join(out).strip()
    except Exception:
        return ""

def _tables_pdfplumber(pdf_bytes: bytes, max_pages: int = 6) -> str:
    try:
        tables_md = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for pi, page in enumerate(pdf.pages[:max_pages], start=1):
                for ti, tbl in enumerate(page.extract_tables() or [], start=1):
                    if not tbl: continue
                    rows = [[(c or "").strip() for c in row] for row in tbl]
                    width = max(len(r) for r in rows)
                    rows = [r + [""]*(width-len(r)) for r in rows]
                    header = rows[0]
                    md = []
                    md.append(f"### Table p{pi}-{ti}")
                    md.append("| " + " | ".join(header) + " |")
                    md.append("| " + " | ".join(["---"]*width) + " |")
                    for r in rows[1:]:
                        md.append("| " + " | ".join(r) + " |")
                    tables_md.append("\n".join(md))
        return "\n\n".join(tables_md).strip()
    except Exception:
        return ""

def extract_text_and_tables(pdf_bytes: bytes) -> str:
    text = _text_pymupdf(pdf_bytes)
    if len(text) < 120:
        alt = _text_pdfminer(pdf_bytes)
        if len(alt) > len(text): text = alt
    if len(text) < 120:
        alt = _text_pypdf(pdf_bytes)
        if len(alt) > len(text): text = alt

    tables_md = _tables_pdfplumber(pdf_bytes, max_pages=6)

    combo = text.strip()
    if tables_md:
        combo = (combo + "\n\n---\n# Extracted Tables (Markdown)\n" + tables_md).strip()
    return _clean(combo)

def candidate_urls(row: pd.Series):
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        for base in ATTACH_BASES:
            cands.append(base + att)
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/"))
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

def fetch_pdf_text_for_df(df_filtered: pd.DataFrame,
                          max_workers=10,
                          request_timeout=25,
                          verbose=False) -> pd.DataFrame:
    work = df_filtered.copy()
    if work.empty:
        work["pdf_url"] = ""
        work["original_text"] = ""
        return work

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": BSE_BASE_PAGE,
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get(BSE_BASE_PAGE, timeout=15)
    except Exception:
        pass

    url_lists = [candidate_urls(row) for _, row in work.iterrows()]
    work["pdf_url"] = ""
    work["original_text"] = ""

    def worker(i, urls):
        for u in urls:
            try:
                r = s.get(u, timeout=request_timeout, allow_redirects=True, stream=False)
                if r.status_code == 200:
                    ctype = (r.headers.get("content-type","") or "").lower()
                    pdf_bytes = r.content
                    head_ok = pdf_bytes[:8].startswith(b"%PDF")
                    if ("pdf" in ctype) or head_ok or u.lower().endswith(".pdf"):
                        txt = extract_text_and_tables(pdf_bytes)
                        if len(txt) >= 10:
                            return i, u, txt
            except Exception:
                continue
        return i, "", ""

    with ThreadPoolExecutor(max_workers=max(2, min(max_workers, 16))) as ex:
        futures = [ex.submit(worker, i, urls) for i, urls in enumerate(url_lists) if urls]
        for fut in as_completed(futures):
            i, u, txt = fut.result()
            if i < len(work.index):
                idx = work.index[i]
                work.at[idx, "pdf_url"] = u
                work.at[idx, "original_text"] = txt

    for col in ["original_text","HEADLINE","NEWSSUB"]:
        if col in work.columns:
            work[col] = work[col].map(_clean)
    return work

# =========================
# Chunking / Embeddings / FAISS
# =========================
def chunk_text(text: str, max_chars=1200, overlap=200):
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return []
    chunks, i = [], 0
    while i < len(s):
        j = min(i + max_chars, len(s))
        k = s.rfind(". ", max(i + max(200, max_chars-300), i), j)
        if k == -1: k = j
        chunks.append(s[i:k].strip())
        if k == j and j >= len(s): break
        i = max(0, k - overlap)
        if i <= 0: i = k
    return [c for c in chunks if len(c) > 20]

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # smaller, cloud-friendly summarizer
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def build_index(df_docs: pd.DataFrame):
    if df_docs.empty:
        return None, None, None
    rows = []
    for i, r in df_docs.iterrows():
        text = str(r.get("original_text") or "")
        meta = {
            "row_id": i,
            "company": str(r.get("SLONGNAME") or ""),
            "headline": str(r.get("HEADLINE") or r.get("NEWSSUB") or ""),
            "date": str(r.get("NEWS_DT") or ""),
            "pdf_url": str(r.get("pdf_url") or ""),
            "scrip": str(r.get("SCRIP_CD") or ""),
        }
        for j, c in enumerate(chunk_text(text, 1200, 200)):
            rows.append({"chunk_text": c, **meta, "chunk_id": f"{i}_{j}"})
    if not rows:
        return None, None, None
    df_chunks = pd.DataFrame(rows)
    embedder = load_embedder()
    X = embedder.encode(df_chunks["chunk_text"].tolist(), show_progress_bar=False, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return df_chunks, index, X

def retrieve(df_chunks, index, query, top_k=6):
    if df_chunks is None or index is None:
        return pd.DataFrame()
    qvec = load_embedder().encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qvec, top_k)
    if I.size == 0:
        return pd.DataFrame()
    hits = df_chunks.iloc[I[0]].copy()
    hits["score"] = D[0]
    return hits

def answer_with_citations(hits: pd.DataFrame, question: str):
    if hits is None or hits.empty:
        return "No relevant context found for this question in the selected range/filters.", []
    context = " ".join(hits["chunk_text"].tolist())
    summarizer = load_summarizer()
    try:
        ans = summarizer(
            f"Answer concisely based only on the context. Note uncertainty if needed.\n\n"
            f"Question: {question}\n\nContext:\n{context}",
            max_length=220, min_length=100, do_sample=False
        )[0]["summary_text"]
    except Exception:
        ans = context[:1000]
    cits = hits[["row_id","company","headline","date","scrip","pdf_url"]].drop_duplicates("row_id").to_dict("records")
    return ans, cits

# =========================
# Storage (append & dedupe)
# =========================
def append_and_save_parquet(df_new: pd.DataFrame, path: Path = PARQUET_PATH):
    if df_new.empty:
        return load_all_docs()
    cols = sorted(df_new.columns)
    if path.exists():
        old = pd.read_parquet(path)
        allc = sorted(set(old.columns) | set(cols))
        old = old.reindex(columns=allc)
        df_new = df_new.reindex(columns=allc)
        key_cols = [c for c in ["ATTACHMENTNAME","NSURL","NEWS_DT","SCRIP_CD"] if c in allc]
        merged = pd.concat([old, df_new], ignore_index=True)
        if key_cols:
            merged["_key"] = merged[key_cols].astype(str).agg("|".join, axis=1)
            merged = merged.drop_duplicates("_key").drop(columns=["_key"])
        merged.to_parquet(path, index=False)
        return merged
    else:
        df_new.to_parquet(path, index=False)
        return df_new

def load_all_docs(path: Path = PARQUET_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

# =========================
# Excel export (in-memory)
# =========================
def df_to_excel_download(df: pd.DataFrame, filename: str = "bse_company_update.xlsx"):
    if df.empty:
        return None
    safe = df.replace({r'[\x00-\x08\x0B-\x0C\x0E-\x1F]': ''}, regex=True).copy()
    if "original_text" in safe.columns:
        safe["original_text"] = safe["original_text"].astype(str).str.slice(0, 32760)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        safe.to_excel(w, index=False, sheet_name="data")
        ws = w.sheets["data"]; wb = w.book
        wrap = wb.add_format({'text_wrap': True})
        ws.set_column(0, len(safe.columns)-1, 28)
        if "original_text" in safe.columns:
            col_idx = safe.columns.get_loc("original_text")
            ws.set_column(col_idx, col_idx, 80, wrap)
    buf.seek(0)
    return buf

# =========================
# Scheduler (Cloud caveat: runs while app is active)
# =========================
def _save_config(d: dict):
    CONFIG_PATH.write_text(json.dumps(d, indent=2))

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}

def daily_pull_job(cfg: dict):
    try:
        now = _now_ist()
        yday = (now - dt.timedelta(days=1)).date()
        start_s = end_s = yday.strftime("%Y%m%d")

        df_all = fetch_bse_announcements_strict(start_s, end_s, verbose=False)
        df_cat = filter_by_category(df_all, "Company Update")

        df_f = apply_company_filters(
            df_cat,
            scrip_codes=cfg.get("scrip_codes",""),
            company_keywords=cfg.get("company_keywords",""),
            exact_company=cfg.get("exact_company", False),
        )

        if df_f.empty:
            return {"status":"ok","fetched":0,"docs":0}

        df_docs = fetch_pdf_text_for_df(
            df_f,
            max_workers=cfg.get("max_workers", 8),
            verbose=False
        )
        df_docs = df_docs[(df_docs["original_text"].str.len() >= 10)]
        if df_docs.empty:
            return {"status":"ok","fetched":len(df_f),"docs":0}

        merged = append_and_save_parquet(df_docs)
        return {"status":"ok","fetched":len(df_f),"docs":len(df_docs),"store_rows":len(merged)}
    except Exception as e:
        return {"status":"error","error":str(e)}

@st.cache_resource(show_spinner=False)
def get_scheduler():
    sched = BackgroundScheduler(timezone=IST)
    sched.start(paused=True)
    return sched

def ensure_daily_job(enabled: bool, hour: int, minute: int, cfg: dict):
    sched = get_scheduler()
    job_id = "daily_bse_pull"
    try:
        j = sched.get_job(job_id)
        if j: sched.remove_job(job_id)
    except Exception:
        pass
    if enabled:
        trigger = CronTrigger(hour=hour, minute=minute, timezone=IST)
        sched.add_job(lambda: daily_pull_job(cfg), trigger=trigger, id=job_id, replace_existing=True)
        sched.resume()
    else:
        sched.pause()

# =========================
# UI
# =========================
st.title("🧠 BSE Company Update – AI Agent (Streamlit Cloud)")
st.caption("Filter by date/ticker/company • Parse PDFs (text + tables) • Ask questions • Export Excel. (OCR disabled on Cloud)")

with st.sidebar:
    st.header("1) Date Range")
    today = _now_ist().date()
    start = st.date_input("From", value=today, max_value=today)
    end = st.date_input("To", value=today, min_value=start, max_value=today)

    st.header("2) Filters")
    scrip_codes = st.text_input("BSE Scrip code(s) (comma/space separated)", value="")
    company_keywords = st.text_input("Company keyword(s) in name (comma separated)", value="")
    exact_company = st.checkbox("Exact company name match", value=False)

    st.header("3) Performance")
    max_workers = st.slider("Parallel downloads", 2, 12, 8)

    run = st.button("Fetch & Build Corpus")

    st.markdown("---")
    st.header("Daily Auto-Pull (while app is running)")
    sched_cfg = _load_config()
    sched_enable = st.checkbox("Enable daily pull", value=sched_cfg.get("sched_enable", False))
    sched_time = st.time_input("Run time (IST)", value=dt.time(7, 0))
    if st.button("Apply Scheduler"):
        cfg = {
            "sched_enable": bool(sched_enable),
            "sched_hour": int(sched_time.hour),
            "sched_min": int(sched_time.minute),
            "scrip_codes": scrip_codes,
            "company_keywords": company_keywords,
            "exact_company": bool(exact_company),
            "max_workers": int(max_workers),
        }
        _save_config(cfg)
        ensure_daily_job(cfg["sched_enable"], cfg["sched_hour"], cfg["sched_min"], cfg)
        st.success(f"Scheduler {'enabled' if cfg['sched_enable'] else 'disabled'} for {sched_time.strftime('%H:%M')} IST.")

# state
for key in ["df_all","df_filtered","df_docs","chunks","faiss"]:
    if key not in st.session_state: st.session_state[key] = pd.DataFrame() if key.startswith("df_") else None

if run:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    with st.spinner("Fetching announcements…"):
        df_all = fetch_bse_announcements_strict(start_s, end_s, verbose=False)
        st.session_state.df_all = df_all

    with st.spinner("Filtering to 'Company Update' + ticker/company…"):
        df_filtered = filter_by_category(st.session_state.df_all, "Company Update")
        df_filtered = apply_company_filters(df_filtered, scrip_codes, company_keywords, exact_company)
        st.session_state.df_filtered = df_filtered

    if st.session_state.df_filtered.empty:
        st.warning("No rows matched filters in this range.")
    else:
        with st.spinner(f"Reading PDFs for {len(st.session_state.df_filtered)} rows…"):
            df_docs = fetch_pdf_text_for_df(
                st.session_state.df_filtered,
                max_workers=max_workers, verbose=False
            )
            df_docs = df_docs[(df_docs["original_text"].str.len() >= 10)]
            st.session_state.df_docs = df_docs

        if st.session_state.df_docs.empty:
            st.error("No readable PDFs found (many PDFs are scanned; OCR is disabled on Streamlit Cloud).")
        else:
            with st.spinner("Building embeddings & index…"):
                chunks, index, _ = build_index(st.session_state.df_docs)
                st.session_state.chunks = chunks
                st.session_state.faiss = index
            st.success(f"Ready! {len(st.session_state.df_docs)} documents, {0 if chunks is None else len(chunks)} chunks indexed.")

st.markdown("### Ask a question")
q = st.text_input("e.g., What acquisitions mention consideration amounts for my filtered companies?")
topk = st.slider("Top results to consider", 3, 12, 6)
if st.button("Answer"):
    if st.session_state.faiss is None or st.session_state.chunks is None:
        st.warning("Please Fetch & Build Corpus first.")
    elif not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and composing answer…"):
            hits = retrieve(st.session_state.chunks, st.session_state.faiss, q, top_k=topk)
            ans, cits = answer_with_citations(hits, q)
        st.subheader("Answer")
        st.write(ans)
        st.subheader("Sources")
        if not cits:
            st.write("No sources.")
        else:
            for c in cits:
                comp = c.get("company","").strip()
                head = c.get("headline","").strip()
                date = c.get("date","").strip()
                scrip = c.get("scrip","").strip()
                url  = c.get("pdf_url","").strip()
                st.markdown(f"- **{comp}** (Scrip: {scrip}) — {head}  \n  _{date}_  •  [PDF]({url})")

st.markdown("---")
st.markdown("### Parsed announcements")
if isinstance(st.session_state.df_docs, pd.DataFrame) and not st.session_state.df_docs.empty:
    cols_show = [c for c in ["SCRIP_CD","SLONGNAME","HEADLINE","NEWS_DT","pdf_url","original_text"] if c in st.session_state.df_docs.columns]
    st.dataframe(st.session_state.df_docs[cols_show].reset_index(drop=True), use_container_width=True, height=360)
    # Excel download
    def df_to_excel_download(df: pd.DataFrame, filename: str = "bse_company_update.xlsx"):
        if df.empty:
            return None
        safe = df.replace({r'[\x00-\x08\x0B-\x0C\x0E-\x1F]': ''}, regex=True).copy()
        if "original_text" in safe.columns:
            safe["original_text"] = safe["original_text"].astype(str).str.slice(0, 32760)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            safe.to_excel(w, index=False, sheet_name="data")
            ws = w.sheets["data"]; wb = w.book
            wrap = wb.add_format({'text_wrap': True})
            ws.set_column(0, len(safe.columns)-1, 28)
            if "original_text" in safe.columns:
                col_idx = safe.columns.get_loc("original_text")
                ws.set_column(col_idx, col_idx, 80, wrap)
        buf.seek(0)
        return buf
    xls = df_to_excel_download(st.session_state.df_docs)
    if xls:
        st.download_button("⬇️ Download Excel", data=xls, file_name="bse_company_update.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.caption("Run a fetch to see parsed documents.")

st.markdown("---")
st.caption("Note: Streamlit Cloud may pause inactive apps; the daily scheduler runs only while the app is active.")
