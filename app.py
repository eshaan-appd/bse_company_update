import os, io, re, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract

import streamlit as st

# ============================
# Page setup
# ============================
st.set_page_config(page_title="BSE Special Situations — M&A/JV Filter", layout="wide")
st.title("📈 BSE Special Situations — M&A/JV Filter")
st.caption("Filters BSE announcements by sub-category (Acquisition, Amalgamation/Merger, Scheme of Arrangement, Joint Venture), extracts PDFs, and summarises (non-LLM or LLM).")

# ============================
# Utilities
# ============================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub("", s) if isinstance(s, str) else s

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _norm_for_match(s):
    s = (s or "").lower()
    s = s.replace("\\", "/")       # guard sloppy backslashes
    s = s.replace("-", "/")        # hyphen to slash for variants
    s = re.sub(r"\s*/\s*", "/", s) # normalize slashes
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ============================
# PDF extraction stack
# ============================
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
                if t:
                    out.append(t)
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
                    if not tbl:
                        continue
                    rows = [[(c or "").strip() for c in row] for row in tbl]
                    width = max(len(r) for r in rows)
                    rows = [r + [""] * (width - len(r)) for r in rows]
                    header = rows[0]
                    md = []
                    md.append(f"### Table p{pi}-{ti}")
                    md.append("| " + " | ".join(header) + " |")
                    md.append("| " + " | ".join(["---"] * width) + " |")
                    for r in rows[1:]:
                        md.append("| " + " | ".join(r) + " |")
                    tables_md.append("\n".join(md))
        return "\n\n".join(tables_md).strip()
    except Exception:
        return ""

def _ocr_first_pages(pdf_bytes: bytes, max_pages: int = 3) -> str:
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=max_pages)
        return "\n".join((pytesseract.image_to_string(im) or "") for im in imgs).strip()
    except Exception:
        return ""

def _extract_text_and_tables(pdf_bytes: bytes, use_ocr=True, ocr_pages=3) -> str:
    text = _text_pymupdf(pdf_bytes)
    if len(text) < 120:
        alt = _text_pdfminer(pdf_bytes)
        if len(alt) > len(text):
            text = alt
    if len(text) < 120:
        alt = _text_pypdf(pdf_bytes)
        if len(alt) > len(text):
            text = alt
    tables_md = _tables_pdfplumber(pdf_bytes, max_pages=6)
    if len(text) < 80 and not tables_md and use_ocr:
        ocr = _ocr_first_pages(pdf_bytes, max_pages=ocr_pages)
        if len(ocr) > len(text):
            text = ocr
    combo = text.strip()
    if tables_md:
        combo = (combo + "\n\n---\n# Extracted Tables (Markdown)\n" + tables_md).strip()
    return _clean(combo)

# ============================
# Attachment URL candidates
# ============================
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
    # de-dupe
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
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

    if verbose:
        st.write(f"PDF candidates for {len(url_lists)} filtered rows; fetching…")

    def worker(i, urls):
        for u in urls:
            try:
                r = s.get(u, timeout=request_timeout, allow_redirects=True, stream=False)
                if r.status_code == 200:
                    ctype = (r.headers.get("content-type", "") or "").lower()
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

    for col in ["original_text", "HEADLINE", "NEWSSUB"]:
        if col in work.columns:
            work[col] = work[col].map(_clean)

    if verbose:
        st.success(f"Filled original_text for {(work['original_text'].str.len() >= 10).sum()} of {len(work)} rows.")
    return work

# ============================
# Fetch BSE announcements (strict)
# ============================
def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   verbose: bool = True,
                                   request_timeout: int = 25) -> pd.DataFrame:
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
            "strType": "C",  # Equity
        }

        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type", "")
            if "application/json" not in ct:
                if verbose:
                    st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break

            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)

            if total is None:
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None

            if verbose:
                msg = f"[variant {v}] page {page}: got {len(table)} (acc {len(rows)})"
                if total is not None:
                    msg += f", total {total}"
                st.write(msg)

            if not table:
                break

            params["pageno"] += 1
            page += 1
            time.sleep(0.3)
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
    df = pd.DataFrame(all_rows, columns=list(all_keys))
    return df

# ============================
# Sub-category filter (STRICT)
# ============================
ALLOWED_SUBCATS = [
    "Acquisition",
    "Amalgamation/Merger",
    "Scheme of Arrangement",
    "Joint Venture",
]

# Accept common variants like "Amalgamation / Merger" or "Amalgamation- Merger"
ALLOWED_NORM = {_norm_for_match(x) for x in ALLOWED_SUBCATS} | {
    _norm_for_match("Amalgamation / Merger"),
    _norm_for_match("Amalgamation - Merger"),
    _norm_for_match("Amalgamation Merger"),
}

def filter_by_subcategory_strict(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    subcol = _first_col(df, ["SUBCATEGORYNAME", "SUB_CATEGORYNAME", "SUBCATEGORY", "SUB_CATEGORY"])
    if not subcol:
        # If sub-category is missing, return empty to be strict
        return df.iloc[0:0].copy()

    def ok(x):
        return _norm_for_match(x) in ALLOWED_NORM

    out = df.loc[df[subcol].map(ok)]
    return out.copy()

# ============================
# Summarisers
# ============================
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
    top_idx_sorted = sorted(top_idx)
    return " ".join([sents[i] for i in top_idx_sorted])

def extract_highlights(text: str, top_k: int = 10) -> list:
    patterns = [
        r"₹\s?[\d,]+(?:\.\d+)?",
        r"Rs\.?\s?[\d,]+(?:\.\d+)?\s?(?:crore|million|bn|billion)?",
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s?%",
        r"\bFY\d{2}\b",
        r"\bFY\d{2}-\d{2}\b",
        r"\bQ[1-4]FY\d{2}\b",
        r"\b[A-Z]{2,}(?:\s[A-Z]{2,})+\b",
    ]
    found = []
    for p in patterns:
        found += re.findall(p, text or "", flags=re.IGNORECASE)
    out, seen = [], set()
    for tok in found:
        key = tok.lower()
        if key not in seen:
            out.append(tok)
            seen.add(key)
        if len(out) >= top_k:
            break
    return out

def llm_summarize(text: str, model: str = "gpt-4o-mini", api_key: str = None, sys_prompt: str = None) -> str:
    if not api_key:
        return "[LLM disabled] Provide OPENAI_API_KEY to enable."
    sys_prompt = sys_prompt or (
        "You are an equity research assistant. Summarize the announcement into crisp bullets:\n"
        "1) What happened, 2) amounts & instruments, 3) timelines & conditions, "
        "4) business/valuation impact, 5) red flags. Keep it 80-120 words, factual."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": text[:16000]}],
                temperature=0.2,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": sys_prompt},
                       {"role": "user", "content": text[:16000]}],
                temperature=0.2,
                max_output_tokens=300,
            )
            out = getattr(resp, "output_text", None)
            return (out or "").strip()
    except Exception as e:
        return f"[LLM error] {e}"

# ============================
# Streamlit UI — Controls
# ============================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    default_start = today - timedelta(days=1)
    start_date = st.date_input("Start date", value=default_start, max_value=today)
    end_date = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    st.subheader("PDF Extraction")
    use_ocr = st.checkbox("Enable OCR fallback", value=True)
    ocr_pages = st.slider("OCR pages (first N)", min_value=1, max_value=5, value=3, step=1)
    max_workers = st.slider("Parallel PDF downloads", 2, 16, value=10)

    st.subheader("Summarization")
    sum_mode = st.radio("Summarizer", ["Non-LLM (TF-IDF)", "LLM (OpenAI-compatible)"], index=0)
    api_key = st.text_input("OPENAI_API_KEY (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model name", value="gpt-4o-mini")
    max_items = st.slider("Max announcements to summarize", 10, 400, value=120, step=10)

    run = st.button("🚀 Fetch & Analyze", type="primary")

def _fmt_date(d: datetime.date) -> str:
    return d.strftime("%Y%m%d")

# ============================
# Pipeline
# ============================
@st.cache_data(show_spinner=False, ttl=60 * 20)
def step1_fetch(start_str: str, end_str: str) -> pd.DataFrame:
    return fetch_bse_announcements_strict(start_str, end_str, verbose=False)

def step2_filter_subcat(df: pd.DataFrame) -> pd.DataFrame:
    df2 = filter_by_subcategory_strict(df)
    return df2

def step3_pdf(df: pd.DataFrame) -> pd.DataFrame:
    return fetch_pdf_text_for_df(
        df, use_ocr=use_ocr, ocr_pages=ocr_pages, max_workers=max_workers, request_timeout=25, verbose=True
    )

def _build_context(row: pd.Series) -> str:
    parts = []
    for k in ["SLONGNAME", "HEADLINE", "NEWSSUB", "NEWS_DT", "CATEGORYNAME", "SUBCATEGORYNAME"]:
        if k in row and str(row[k] or "").strip():
            parts.append(f"{k}: {row[k]}")
    if row.get("original_text"):
        parts.append(row["original_text"][:20000])
    return "\n".join(parts)

def step4_summarize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        return work
    if "NEWS_DT" in work.columns:
        try:
            work["_dt"] = pd.to_datetime(work["NEWS_DT"], errors="coerce")
            work = work.sort_values("_dt", ascending=False).drop(columns=["_dt"])
        except Exception:
            pass
    if len(work) > max_items:
        work = work.head(max_items)

    sums, highs, ctxs = [], [], []
    use_llm = (sum_mode.startswith("LLM") and api_key.strip() != "")
    for _, row in work.iterrows():
        ctx = _build_context(row)
        ctxs.append(ctx)
        if use_llm:
            s = llm_summarize(ctx, model=model.strip(), api_key=api_key.strip())
        else:
            body = row.get("original_text") or " ".join([str(row.get("HEADLINE") or ""), str(row.get("NEWSSUB") or "")])
            s = summarize_extractive_tfidf(body, max_sentences=3)
        sums.append(s)
        highs.append(", ".join(extract_highlights(ctx, top_k=8)))

    work["summary"] = sums
    work["highlights"] = highs
    work["context_used"] = ctxs
    return work

def step5_digest_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "# No matching sub-category announcements found."
    cols = df.columns
    nm = "SLONGNAME" if "SLONGNAME" in cols else _first_col(df, ["SNAME", "SC_NAME", "COMPANYNAME"]) or "Company"
    dtcol = "NEWS_DT" if "NEWS_DT" in cols else None
    subcol = _first_col(df, ["SUBCATEGORYNAME", "SUB_CATEGORYNAME", "SUBCATEGORY", "SUB_CATEGORY"])

    lines = ["# BSE Special Situations Digest", ""]
    for _, r in df.iterrows():
        cmp = str(r.get(nm) or "").strip()
        hd = str(r.get("HEADLINE") or "").strip()
        dt = str(r.get(dtcol) or "").strip() if dtcol else ""
        url = str(r.get("pdf_url") or r.get("NSURL") or "").strip()
        sub = str(r.get(subcol) or "").strip() if subcol else ""
        summ = str(r.get("summary") or "").strip()
        lines.append(f"- **{cmp}** — {dt} — _{sub}_")
        if hd:
            lines.append(f"  - *{hd}*")
        if url:
            lines.append(f"  - PDF: {url}")
        if summ:
            lines.append(f"  - {summ}")
    lines.append("")
    return "\n".join(lines)

# ============================
# Run
# ============================
if run:
    start_str, end_str = _fmt_date(start_date), _fmt_date(end_date)
    with st.status("Fetching announcements from BSE…", expanded=True) as status:
        df_raw = step1_fetch(start_str, end_str)
        if df_raw.empty:
            st.error("No rows returned from BSE. Try another date or re-run.")
            status.update(label="No data", state="error")
        else:
            st.write(f"Fetched **{len(df_raw)}** rows.")
            status.update(label="Fetched announcements", state="running")

    if not df_raw.empty:
        with st.status("Filtering by sub-category (Acquisition | Amalgamation/Merger | Scheme of Arrangement | Joint Venture)…", expanded=True) as status:
            df_sub = step2_filter_subcat(df_raw)
            st.write(f"Matched **{len(df_sub)}** rows in required sub-categories.")
            status.update(label="Sub-category filtered", state="running")

        if df_sub.empty:
            st.warning("No announcements in the selected sub-categories for this window.")
        else:
            with st.status("Downloading & extracting PDFs…", expanded=True) as status:
                df_pdf = step3_pdf(df_sub)
                st.write("Sample PDF URL(s):", df_pdf["pdf_url"].replace("", np.nan).dropna().head(3).tolist())
                status.update(label="PDFs extracted", state="running")

            with st.status("Summarizing…", expanded=True) as status:
                df_sum = step4_summarize(df_pdf)
                status.update(label="Summarized", state="complete")

            st.subheader("📑 Results")
            show_cols = [c for c in [
                "NEWS_DT","SLONGNAME","HEADLINE","CATEGORYNAME","SUBCATEGORYNAME",
                "highlights","summary","pdf_url","NSURL"
            ] if c in df_sum.columns]
            st.dataframe(df_sum[show_cols], use_container_width=True, hide_index=True)

            csv_bytes = df_sum[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_bytes,
                               file_name=f"mna_jv_{start_str}_{end_str}.csv", mime="text/csv")

            digest_md = step5_digest_md(df_sum)
            st.download_button("⬇️ Download Markdown Digest", data=digest_md.encode("utf-8"),
                               file_name=f"digest_{start_str}_{end_str}.md", mime="text/markdown")

            with st.expander("📄 Preview Digest (Markdown)"):
                st.markdown(digest_md)

else:
    st.info("Pick your date range and click **Fetch & Analyze**. Only the sub-categories Acquisition, Amalgamation/Merger, Scheme of Arrangement, and Joint Venture are considered.")
