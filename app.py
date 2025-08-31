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
st.set_page_config(page_title="BSE Company Update — M&A/JV Filter (NLP only)", layout="wide")
st.title("📈 BSE Company Update — M&A / Merger / Scheme / JV (NLP)")
st.caption("Fetch BSE announcements → filter by Company Update + (Acquisition | Amalgamation/Merger | Scheme of Arrangement | Joint Venture) → read ALL pages → summarize with offline NLP bullets.")

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
# PDF extraction helpers (read ALL pages)
# =========================================
def _text_pymupdf_all(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join((p.get_text("text") or "") for p in doc).strip()
    except Exception:
        return ""

def _text_pdfminer_all(pdf_bytes: bytes) -> str:
    try:
        return (pdfminer_extract_text(io.BytesIO(pdf_bytes)) or "").strip()
    except Exception:
        return ""

def _text_pypdf_all(pdf_bytes: bytes) -> str:
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

def _tables_pdfplumber_all(pdf_bytes: bytes) -> str:
    """Extract tables from ALL pages as Markdown."""
    try:
        tables_md = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for pi, page in enumerate(pdf.pages, start=1):
                tbls = page.extract_tables() or []
                for ti, tbl in enumerate(tbls, start=1):
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

def _ocr_all_pages(pdf_bytes: bytes, max_pages: int | None = None, dpi: int = 200) -> str:
    """OCR the full document (or up to max_pages) as a last resort."""
    if not (_HAS_PDF2IMAGE and _HAS_PYTESS and _HAS_POPPLER and _HAS_TESS_BIN):
        return ""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        if max_pages:
            imgs = imgs[:max_pages]
        return "\n".join((pytesseract.image_to_string(im) or "") for im in imgs).strip()
    except Exception:
        return ""

def extract_pdf_fulltext(pdf_bytes: bytes, use_ocr=True, ocr_max_pages: int | None = None) -> str:
    """
    Read ALL pages with multiple strategies and include ALL tables as Markdown.
    If digital text is scarce, OCR the whole doc (or up to ocr_max_pages).
    """
    text = _text_pymupdf_all(pdf_bytes)
    alt = _text_pdfminer_all(pdf_bytes)
    if len(alt) > len(text): text = alt
    alt = _text_pypdf_all(pdf_bytes)
    if len(alt) > len(text): text = alt

    tables_md = _tables_pdfplumber_all(pdf_bytes)

    # If still very short, attempt OCR on ALL pages (or limited)
    if use_ocr and len(text) < 300:
        ocr_text = _ocr_all_pages(pdf_bytes, max_pages=ocr_max_pages)
        if len(ocr_text) > len(text):
            text = ocr_text

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
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

def fetch_pdf_text_for_df(
    df_filtered: pd.DataFrame,
    use_ocr: bool = True,
    ocr_max_pages: int | None = None,
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
                        txt = extract_pdf_fulltext(pdf_bytes, use_ocr=use_ocr, ocr_max_pages=ocr_max_pages)
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
    """Fetches raw announcements, then filters:
    Category='Company Update' AND subcategory contains any:
    Acquisition | Amalgamation / Merger | Scheme of Arrangement | Joint Venture
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
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose: st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try: total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception: total = None
            if not table: break
            params["pageno"] += 1; page += 1; time.sleep(0.25)
            if total and len(rows) >= total: break
        if rows:
            all_rows = rows; break

    if not all_rows: return pd.DataFrame()

    # Build DF
    all_keys = set()
    for r in all_rows: all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Filter: Company Update + M&A/Merger/Scheme/JV in subcategory fields
    def filter_announcements(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
        if df_in.empty: return df_in.copy()
        cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
        if not cat_col: return df_in.copy()
        df2 = df_in.copy()
        df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
        return df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])

    df_filtered = filter_announcements(df, category_filter="Company Update")
    df_filtered = df_filtered.loc[
        df_filtered.filter(["NEWSSUB","SUBCATEGORY","SUBCATEGORYNAME","NEWS_SUBCATEGORY","NEWS_SUB"], axis=1)
        .astype(str)
        .apply(lambda col: col.str.contains(r"(Acquisition|Amalgamation\s*/\s*Merger|Scheme of Arrangement|Joint Venture)", case=False, na=False))
        .any(axis=1)
    ]
    return df_filtered

# =========================================
# NLP Summarizer (non-LLM)
# =========================================
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")

KEY_VERBS = re.compile(r"\b(acquir|purchase|merg|amalgamat|scheme|demerg|slump\s+sale|joint\s+venture|jv|arrangement|investment|subscribe|allot)\w*\b", re.I)
MONEY_RX  = re.compile(r"(₹\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr|lakh|mn|million|bn|billion)?|Rs\.?\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr|mn|million|bn|billion)?)", re.I)
PCT_RX    = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%\b")
DATE_RX   = re.compile(r"\b(?:\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|FY\d{2}(?:-\d{2})?|Q[1-4]FY\d{2})\b", re.I)
APPROVAL_RX = re.compile(r"\b(SEBI|NCLT|CCI|RBI|stock\s+exchange|shareholder|board|creditor)s?\b", re.I)
CONSID_RX = re.compile(r"\b(cash|equity\s+shares?|preference\s+shares?|share\s+swap|debentures?|NCDs?|warrants?)\b", re.I)
COMPANY_LIKE_RX = re.compile(r"\b([A-Z][A-Za-z0-9&\.\-]*(?:\s+[A-Z][A-Za-z0-9&\.\-]*)*\s+(?:Limited|Ltd\.?|LLP|Private\s+Limited|Pvt\.?\s+Ltd\.?|Inc\.?|LLC|PLC|AG|SA))\b")

from sklearn.feature_extraction.text import TfidfVectorizer

def _split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    sents = _SENT_SPLIT.split(text)
    if len(sents) < 2:
        sents = re.split(r"\s*[\n;]\s*", text)
    return [s for s in sents if len(s) > 20]

def _rank_sentences_tfidf(sents: list[str]) -> list[int]:
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vec.fit_transform(sents)
    base = np.asarray(X.sum(axis=1)).ravel()
    # Boost for sentences containing key verbs, numbers, approvals, etc.
    boosts = []
    for s in sents:
        b = 0.0
        if KEY_VERBS.search(s): b += 1.2
        if MONEY_RX.search(s):  b += 0.8
        if PCT_RX.search(s):    b += 0.6
        if APPROVAL_RX.search(s): b += 0.5
        if DATE_RX.search(s):   b += 0.4
        if CONSID_RX.search(s): b += 0.4
        boosts.append(b)
    boosts = np.array(boosts)
    scores = base + boosts
    return list(np.argsort(scores)[::-1])

def _extract_highlights(text: str, top_k: int = 10) -> list[str]:
    bits = []
    for rx in [MONEY_RX, PCT_RX, DATE_RX, APPROVAL_RX, CONSID_RX]:
        bits += [m.group(0).strip() for m in rx.finditer(text or "")]
    # de-dupe while preserving order
    seen, out = set(), []
    for b in bits:
        k = b.lower()
        if k not in seen:
            out.append(b); seen.add(k)
        if len(out) >= top_k: break
    return out

def _extract_parties(text: str, self_company: str) -> list[str]:
    names = [m.group(1).strip() for m in COMPANY_LIKE_RX.finditer(text or "")]
    names = [n for n in names if not self_company or _norm(n).lower() != _norm(self_company).lower()]
    # de-dupe preserving order
    seen, out = set(), []
    for n in names:
        k = _norm(n).lower()
        if k not in seen:
            out.append(n); seen.add(k)
    return out[:6]  # limit

def summarize_to_bullets(full_text: str, company: str, headline: str, subcat: str) -> str:
    """
    Offline NLP bullets:
      - Company
      - What it’s about (uses headline or best 'deal' sentence)
      - Key features (top facts: money/percent/dates/approvals/consideration + top ranked sentences)
      - Parties/counterparties
    """
    text = full_text or ""
    sents = _split_sentences(text)
    ranked_idx = _rank_sentences_tfidf(sents)
    best_deal_sent = next((sents[i] for i in ranked_idx if KEY_VERBS.search(sents[i])), headline or "")
    top_fact_sents = []
    for i in ranked_idx:
        s = sents[i]
        if any(rx.search(s) for rx in [MONEY_RX, PCT_RX, APPROVAL_RX, DATE_RX, CONSID_RX]):
            top_fact_sents.append(s)
        if len(top_fact_sents) >= 5:
            break

    highlights = _extract_highlights(text, top_k=10)
    parties = _extract_parties(text, self_company=company)

    # Build Markdown bullets
    bullets = []
    bullets.append(f"- **Company:** {company or 'Unknown'}")
    bullets.append(f"- **Announcement type:** {subcat or 'N/A'}")
    about_line = headline or (best_deal_sent.strip()[:240] + ("…" if len(best_deal_sent) > 240 else ""))
    bullets.append(f"- **What it’s about:** {about_line}")
    if parties:
        bullets.append(f"- **Involved parties / entities:** " + "; ".join(parties))
    if highlights:
        bullets.append(f"- **Key data points:** " + "; ".join(highlights))
    if top_fact_sents:
        bullets.append(f"- **Key features:**")
        for s in top_fact_sents[:5]:
            bullets.append(f"  - {s.strip()}")
    return "\n".join(bullets)

# =========================================
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    use_ocr = st.checkbox("Enable OCR fallback (all pages if needed)", value=True)
    ocr_cap  = st.number_input("OCR max pages (0 = all)", min_value=0, max_value=2000, value=0, step=1)
    ocr_max_pages = None if ocr_cap == 0 else int(ocr_cap)

    max_workers = st.slider("Parallel PDF downloads", 2, 16, 10)
    max_items = st.slider("Max announcements to summarize", 5, 200, 60, step=5)

    run = st.button("🚀 Fetch & Analyze", type="primary")

# =========================================
# Run pipeline (fetch → PDFs → NLP bullets)
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
        if len(df_hits) > max_items:
            df_hits = df_hits.head(max_items)

        with st.status("Downloading & reading ALL PDF pages…", expanded=True):
            df_pdf = fetch_pdf_text_for_df(
                df_hits,
                use_ocr=use_ocr,
                ocr_max_pages=ocr_max_pages,
                max_workers=max_workers,
                request_timeout=25,
                verbose=True
            )

        # Build NLP bullet summaries
        st.subheader("📑 Summaries")
        nm = _first_col(df_pdf, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
        subcol = _first_col(df_pdf, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"

        cnt = 0
        for _, r in df_pdf.iterrows():
            company = str(r.get(nm) or "").strip()
            headline = str(r.get("HEADLINE") or "").strip()
            subcat = str(r.get(subcol) or "").strip()
            fulltext = r.get("original_text") or " ".join([headline, str(r.get("NEWSSUB") or "")])
            pdf_url = str(r.get("pdf_url") or r.get("NSURL") or "").strip()
            dt = str(r.get("NEWS_DT") or "").strip()

            bullets_md = summarize_to_bullets(fulltext, company, headline, subcat)

            with st.expander(f"{company or 'Unknown'} — {dt}  •  {subcat or 'N/A'}"):
                if headline:
                    st.markdown(f"**Headline:** {headline}")
                if pdf_url:
                    st.markdown(f"[PDF link]({pdf_url})")
                st.markdown(bullets_md)
            cnt += 1

        if cnt == 0:
            st.info("No items to show after PDF processing.")
else:
    st.info("Pick your date range and click **Fetch & Analyze**. This version uses offline NLP (no OpenAI) and shows bullet summaries directly on the page.")
