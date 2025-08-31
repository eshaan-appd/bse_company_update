import os, io, re, time, shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader

# --- Optional OCR (kept simple and safe) ---
try:
    from pdf2image import convert_from_bytes
    _HAS_PDF2IMAGE = True
except Exception:
    convert_from_bytes = None
    _HAS_PDF2IMAGE = False

try:
    import pytesseract
    _HAS_PYTESS = True
except Exception:
    pytesseract = None
    _HAS_PYTESS = False

_HAS_POPPLER = shutil.which("pdfinfo") is not None
_HAS_TESS_BIN = shutil.which("tesseract") is not None

import streamlit as st

# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="BSE Company Update — M&A/JV Filter", layout="wide")
st.title("📈 BSE Company Update — M&A / Merger / Scheme / JV")
st.caption("Fetch BSE announcements → filter by Company Update + (Acquisition | Amalgamation/Merger | Scheme of Arrangement | Joint Venture) → extract PDFs → summarize (non-LLM).")

# =========================================
# Small utilities
# =========================================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

# =========================================
# PDF extraction helpers (simple + robust)
# =========================================
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

def _ocr_first_pages(pdf_bytes: bytes, max_pages: int = 3) -> str:
    # Keep OCR optional and quiet
    if not (_HAS_PDF2IMAGE and _HAS_PYTESS and _HAS_POPPLER and _HAS_TESS_BIN):
        return ""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=max_pages)
        return "\n".join((pytesseract.image_to_string(im) or "") for im in imgs).strip()
    except Exception:
        return ""

def _extract_text_and_tables(pdf_bytes: bytes, use_ocr=True, ocr_pages=3) -> str:
    text = _text_pymupdf(pdf_bytes)
    if len(text) < 120:
        alt = _text_pdfminer(pdf_bytes)
        if len(alt) > len(text): text = alt
    if len(text) < 120:
        alt = _text_pypdf(pdf_bytes)
        if len(alt) > len(text): text = alt

    tables_md = _tables_pdfplumber(pdf_bytes, max_pages=6)
    if len(text) < 80 and not tables_md and use_ocr:
        ocr = _ocr_first_pages(pdf_bytes, max_pages=ocr_pages)
        if len(ocr) > len(text): text = ocr

    combo = text.strip()
    if tables_md:
        combo = (combo + "\n\n---\n# Extracted Tables (Markdown)\n" + tables_md).strip()
    return _clean(combo)

# =========================================
# Attachment URL candidates
# =========================================
def _candidate_urls(row):
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/"))
    # de-dup
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

def fetch_pdf_text_for_df(
    df_filtered: pd.DataFrame,
    use_ocr: bool = True,
    ocr_pages: int = 3,
    max_workers: int = 10,
    request_timeout: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    work = df_filtered.copy()
    if work.empty:
        work["pdf_url"] = ""
        work["original_text"] = ""
        return work

    base_page = "https://www.bseindia.com/corporates/ann.html"
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": base_page,
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    url_lists = [_candidate_urls(row) for _, row in work.iterrows()]
    work["pdf_url"] = ""
    work["original_text"] = ""

    if verbose: st.write(f"PDF candidates for {len(url_lists)} filtered rows; fetching…")

    def worker(i, urls):
        for u in urls:
            try:
                r = s.get(u, timeout=request_timeout, allow_redirects=True, stream=False)
                if r.status_code == 200:
                    ctype = (r.headers.get("content-type","") or "").lower()
                    pdf_bytes = r.content
                    head_ok = pdf_bytes[:8].startswith(b"%PDF")
                    if ("pdf" in ctype) or head_ok or u.lower().endswith(".pdf"):
                        txt = _extract_text_and_tables(pdf_bytes, use_ocr=use_ocr, ocr_pages=ocr_pages)
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

    if verbose:
        st.success(f"Filled original_text for {(work['original_text'].str.len()>=10).sum()} of {len(work)} rows.")
    return work

# =========================================
# BSE fetch — strict; returns filtered DF
# =========================================
def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   verbose: bool = True,
                                   request_timeout: int = 25) -> pd.DataFrame:
    """
    Fetches raw announcements, then applies your exact filters:
    (1) Category = 'Company Update'
    (2) Sub-category contains any of: Acquisition | Amalgamation / Merger | Scheme of Arrangement | Joint Venture
    """
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd

    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    try:
        s.get(base_page, timeout=15)
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
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose:
                    st.warning(f"[variant {v}] non-JSON response on page {page} (ct={ct}).")
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

    # Build DF
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # ======== YOUR EXACT FILTERS (copy-paste friendly) ========
    # Helper: BSE category filter
    def filter_announcements(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
        if df_in.empty: return df_in.copy()
        cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
        if not cat_col: return df_in.copy()
        df2 = df_in.copy()
        df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
        out = df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])
        return out

    # 1) Only "Company Update"
    df_filtered = filter_announcements(df, category_filter="Company Update")

    # 2) Sub-category text match across common fields
    df_filtered = df_filtered.loc[
        df_filtered.filter(["NEWSSUB","SUBCATEGORY","SUBCATEGORYNAME","NEWS_SUBCATEGORY","NEWS_SUB"], axis=1)
        .astype(str)
        .apply(lambda col: col.str.contains(r"(Acquisition|Amalgamation\s*/\s*Merger|Scheme of Arrangement|Joint Venture)", case=False, na=False))
        .any(axis=1)
    ]
    # ==========================================================

    return df_filtered

# =========================================
# Simple summarizer (non-LLM TF-IDF)
# =========================================
def _split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    sents = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", text)
    if len(sents) < 2:
        sents = re.split(r"\s*[\n;]\s*", text)
    return [s for s in sents if len(s) > 20]

def summarize_extractive_tfidf(text: str, max_sentences: int = 3) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = _split_sentences(text)
    if not sents:
        return (text or "")[:400]
    if len(sents) <= max_sentences:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(sents)
    scores = np.asarray(X.sum(axis=1)).ravel()
    top_idx = np.argsort(scores)[::-1][:max_sentences]
    return " ".join([sents[i] for i in sorted(top_idx)])

# =========================================
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    use_ocr = st.checkbox("Enable OCR fallback", value=True)
    if use_ocr and (not _HAS_PDF2IMAGE or not _HAS_PYTESS or not _HAS_POPPLER or not _HAS_TESS_BIN):
        st.info("OCR dependencies not found; OCR will be skipped.")
    ocr_pages = st.slider("OCR pages (first N)", 1, 5, 3)
    max_workers = st.slider("Parallel PDF downloads", 2, 16, 10)

    run = st.button("🚀 Fetch & Analyze", type="primary")

# =========================================
# Run pipeline (fetch → PDFs → summarize)
# =========================================
def _fmt(d: datetime.date) -> str: return d.strftime("%Y%m%d")

if run:
    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_hits = fetch_bse_announcements_strict(start_str, end_str, verbose=False)
        st.write(f"Matched rows after filters: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No matching announcements in this window.")
    else:
        with st.status("Downloading & extracting PDFs…", expanded=True):
            df_pdf = fetch_pdf_text_for_df(
                df_hits, use_ocr=use_ocr, ocr_pages=ocr_pages,
                max_workers=max_workers, request_timeout=25, verbose=True
            )

        with st.status("Summarizing (TF-IDF)…", expanded=True):
            summaries = []
            for _, r in df_pdf.iterrows():
                body = r.get("original_text") or " ".join([str(r.get("HEADLINE") or ""), str(r.get("NEWSSUB") or "")])
                summaries.append(summarize_extractive_tfidf(body, max_sentences=3))
            df_pdf["summary"] = summaries

        # Show & download
        show_cols = [c for c in [
            "NEWS_DT","SLONGNAME","HEADLINE","CATEGORYNAME","SUBCATEGORYNAME",
            "NEWSSUB","pdf_url","NSURL","summary"
        ] if c in df_pdf.columns]
        st.subheader("📑 Results")
        st.dataframe(df_pdf[show_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download CSV",
            data=df_pdf[show_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"company_update_mna_jv_{start_str}_{end_str}.csv",
            mime="text/csv"
        )
else:
    st.info("Pick your date range and click **Fetch & Analyze**.")
