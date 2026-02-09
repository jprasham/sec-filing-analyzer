import os, re, json, time, traceback
from datetime import datetime, timedelta, date, timezone
from typing import Dict, Optional, List, Any, Tuple
from flask import Flask, request, jsonify, Response

import requests as http_req

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ── timezone helpers ──────────────────────────────────────
ET = ZoneInfo("America/New_York") if ZoneInfo else None
UTC_TZ = ZoneInfo("UTC") if ZoneInfo else timezone.utc

# ── SEC constants ─────────────────────────────────────────
SEC_UA = os.environ.get("SEC_USER_AGENT", "SECAnalyzer admin@example.com")
SEC_H = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
SEC_AH = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}

# ── Flask app ─────────────────────────────────────────────
app = Flask(__name__)

# ── Gemini setup ──────────────────────────────────────────
_gemini_model = None

def get_gemini_model():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not genai:
        raise ValueError("google-generativeai package not available")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview",
        system_instruction=(
            "You are a disciplined, data-driven investment analyst. "
            "Do not fabricate, estimate, or assume any facts, numbers, or figures. "
            "If data is unavailable, use the string \"null\" where instructed. "
            "When instructed to output STRICT JSON, output only valid JSON with no extra text."
        ),
    )
    return _gemini_model


def query_gemini(prompt: str) -> str:
    model = get_gemini_model()
    resp = model.generate_content(prompt)
    return getattr(resp, "text", str(resp))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEC HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ticker_cache: Dict[str, Dict[str, str]] = {}


def load_ticker_metadata() -> Dict[str, Dict[str, str]]:
    global _ticker_cache
    if _ticker_cache:
        return _ticker_cache
    url = "https://www.sec.gov/files/company_tickers.json"
    r = http_req.get(url, headers=SEC_AH, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik_int = entry.get("cik_str")
        name = str(entry.get("title", "")).strip()
        if ticker and cik_int is not None:
            out[ticker] = {"cik": f"{int(cik_int):010d}", "name": name or ticker}
    _ticker_cache = out
    return out


def get_submissions(cik: str) -> Optional[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = http_req.get(url, headers=SEC_H, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_dt(s):
    if not s:
        return None
    s = str(s).strip()
    is_utc = s.endswith("Z")
    if is_utc:
        s = s[:-1]
    if "." in s:
        s = s.split(".", 1)[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if is_utc:
                dt = dt.replace(tzinfo=UTC_TZ)
                return dt.astimezone(ET) if ET else dt
            return dt.replace(tzinfo=ET) if ET else dt
        except Exception:
            pass
    return None


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def filing_contains_item_202(html: str) -> bool:
    text = html_to_text(html)
    if re.search(r"\bitem\s*2\.?\s*02\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"results of operations and financial condition", text, flags=re.IGNORECASE):
        return True
    return False


def _build_url(cik, accession, filename):
    cik_clean = str(int(cik))
    acc_nd = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_nd}/{filename}"


def fetch_filing_html(cik, accession, primary_doc):
    url = _build_url(cik, accession, primary_doc)
    r = http_req.get(url, headers=SEC_AH, timeout=25)
    r.raise_for_status()
    return r.text


def fetch_index_json(cik, accession):
    cik_clean = str(int(cik))
    acc_nd = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_nd}/index.json"
    try:
        r = http_req.get(url, headers=SEC_AH, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def choose_ex99_1(index_json, primary_doc=None):
    items = (index_json or {}).get("directory", {}).get("item", [])
    if not isinstance(items, list):
        return None
    primary_low = (primary_doc or "").lower().strip()
    candidates = []
    for it in items:
        name = str(it.get("name", "")).strip()
        if not name:
            continue
        low = name.lower()
        if not any(low.endswith(ext) for ext in (".htm", ".html", ".txt")):
            continue
        if primary_low and low == primary_low:
            continue
        candidates.append(name)
    if not candidates:
        return None

    def score(fn):
        f = fn.lower()
        s = 0
        if re.search(r"(ex99[\._-]?1|exhibit[\s\._-]?99[\._-]?1|99[\._-]?1)", f):
            s += 1000
        if re.search(r"(earnings|press|release)", f):
            s += 400
        if re.search(r"(exhibit|ex99|ex-99|ex_99)", f):
            s += 250
        if f.endswith(".htm") or f.endswith(".html"):
            s += 50
        if re.search(r"(form8k|8-k|10q|10-q|index)", f):
            s -= 100
        return s

    ranked = sorted(candidates, key=lambda x: (-score(x), len(x)))
    return ranked[0]


def fetch_effective_8k_html(cik, accession, primary_doc):
    idx = fetch_index_json(cik, accession)
    if idx:
        ex_fn = choose_ex99_1(idx, primary_doc=primary_doc)
        if ex_fn:
            url = _build_url(cik, accession, ex_fn)
            try:
                r = http_req.get(url, headers=SEC_AH, timeout=25)
                r.raise_for_status()
                return r.text
            except Exception:
                pass
    return fetch_filing_html(cik, accession, primary_doc)


def get_qualifying_filings(cik: str, days: int = 30) -> List[dict]:
    data = get_submissions(cik)
    if not data:
        return []
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    reports = recent.get("reportDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    n = min(len(forms), len(dates), len(accessions), len(docs))
    cutoff = (datetime.now(ET) if ET else datetime.utcnow()).date() - timedelta(days=days)
    results = []
    for i in range(n):
        fd = _parse_date(dates[i])
        if not fd or fd < cutoff:
            continue
        form = str(forms[i]).strip()
        if form not in ("10-Q", "8-K"):
            continue
        acc = accessions[i]
        pdoc = docs[i]
        rpt = reports[i] if i < len(reports) else None
        adt = acceptances[i] if i < len(acceptances) else None
        rec = {
            "form": form,
            "accession_number": acc,
            "primary_document": pdoc,
            "filing_date": dates[i],
            "report_period": rpt,
            "acceptanceDateTime": adt,
        }
        if form == "10-Q":
            results.append(rec)
        elif form == "8-K":
            try:
                html = fetch_filing_html(cik, acc, pdoc)
                time.sleep(0.11)
                if filing_contains_item_202(html):
                    results.append(rec)
            except Exception:
                continue
        if len(results) >= 5:
            break
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCORING ENGINE (identical to Streamlit version)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_value(val_str, unit_str):
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
    s = str(val_str).strip()
    if not s or s.lower() == "null":
        return None
    clean = re.sub(r"[^\d\.\(\)\-]", "", s)
    is_neg = ("(" in clean and ")" in clean) or clean.startswith("-")
    clean = clean.replace("(", "").replace(")", "").replace("-", "")
    try:
        val = float(clean)
    except ValueError:
        return None
    if is_neg:
        val = -val
    u = str(unit_str or "ones").lower()
    mult = 1
    if "thous" in u:
        mult = 1_000
    elif "mill" in u:
        mult = 1_000_000
    elif "bill" in u:
        mult = 1_000_000_000
    return val * mult


def normalize_data(raw_json):
    meta = raw_json.get("meta", {}) or {}
    data = raw_json.get("data", {}) or {}
    unit = meta.get("reporting_unit", "ones")
    normalized = {}
    for category, values in data.items():
        normalized[category] = {}
        if not isinstance(values, dict):
            continue
        for k, v in values.items():
            normalized[category][k] = _parse_value(v, unit)
    return normalized


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(cur, prev):
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


def compute_scorecard(data_json):
    m = normalize_data(data_json)
    f = data_json.get("flags", {}) or {}
    base = 50
    rev_c = m.get("revenue", {}).get("current")
    rev_p = m.get("revenue", {}).get("prior")
    gp_c = m.get("gross_profit", {}).get("current")
    gp_p = m.get("gross_profit", {}).get("prior")
    cos_c = m.get("cost_of_sales", {}).get("current")
    cos_p = m.get("cost_of_sales", {}).get("prior")
    if gp_c is None and rev_c is not None and cos_c is not None:
        gp_c = rev_c - cos_c
    if gp_p is None and rev_p is not None and cos_p is not None:
        gp_p = rev_p - cos_p
    op_inc = m.get("operating_income", {}).get("current")
    ni = m.get("net_income_gaap", {}).get("current")
    ocf = m.get("ocf", {}).get("current")
    capex = m.get("capex", {}).get("current")
    sbc = m.get("sbc_expense", {}).get("current")
    interest = m.get("interest_expense", {}).get("current")
    rpo_c = m.get("rpo", {}).get("current")
    rpo_p = m.get("rpo", {}).get("prior")
    ar_c = m.get("accounts_receivable", {}).get("current")
    ar_p = m.get("accounts_receivable", {}).get("prior")

    gm_c = _safe_div(gp_c, rev_c)
    gm_p = _safe_div(gp_p, rev_p)
    gm_bps = (gm_c - gm_p) * 10000.0 if gm_c is not None and gm_p is not None else None
    rev_g = _growth(rev_c, rev_p)
    rpo_g = _growth(rpo_c, rpo_p)
    ar_g = _growth(ar_c, ar_p)
    fcf = (ocf + capex) if ocf is not None and capex is not None else None
    ocf_ni = _safe_div(ocf, ni) if ni not in (None, 0) else None
    int_opinc = _safe_div(abs(interest) if interest else None, op_inc) if op_inc not in (None, 0) else None
    sbc_ocf = _safe_div(sbc, ocf) if ocf not in (None, 0) else None

    pts = {"base": base, "snowflake": 0, "amazon": 0, "margin_rocket": 0,
           "oracle": 0, "target": 0, "zombie": 0, "sbc_death_ratio": 0,
           "channel_stuffing": 0, "quality_trap": 0, "gain_waiver": 0, "kitchen_sink": 0}
    notes = []

    # Bonuses
    if rpo_g is not None and rev_g is not None:
        if rpo_g >= (rev_g + 0.15):
            pts["snowflake"] = 20; notes.append("PASS: Snowflake G1 (+20)")
        elif rpo_g >= (rev_g + 0.05):
            pts["snowflake"] = 10; notes.append("PASS: Snowflake G2 (+10)")
    if fcf is not None and ocf is not None and ni is not None:
        if fcf < 0 and ocf > ni:
            pts["amazon"] = 10; notes.append("PASS: Amazon Exception (+10)")
    if gm_bps is not None:
        if gm_bps >= 300:
            pts["margin_rocket"] = 15; notes.append(f"PASS: Margin Rocket G1 (+15) | {gm_bps:.0f} bps")
        elif gm_bps >= 150:
            pts["margin_rocket"] = 5; notes.append(f"PASS: Margin Rocket G2 (+5) | {gm_bps:.0f} bps")

    # Penalties
    if ocf_ni is not None:
        if ocf_ni < 0.9:
            pts["oracle"] = -25; notes.append(f"FAIL: Oracle G2 (-25) | OCF/NI={ocf_ni:.2f}")
        elif ocf_ni < 1.0:
            pts["oracle"] = -10; notes.append(f"FAIL: Oracle G1 (-10) | OCF/NI={ocf_ni:.2f}")
    if gm_bps is not None:
        if gm_bps <= -100:
            pts["target"] = -25; notes.append(f"FAIL: Target G2 (-25) | {gm_bps:.0f} bps")
        elif gm_bps <= -50:
            pts["target"] = -10; notes.append(f"FAIL: Target G1 (-10) | {gm_bps:.0f} bps")
    if int_opinc is not None and int_opinc >= 0:
        if int_opinc > 0.40:
            pts["zombie"] = -25; notes.append(f"FAIL: Zombie G2 (-25) | Int/OpInc={int_opinc:.2f}")
        elif int_opinc >= 0.20:
            pts["zombie"] = -10; notes.append(f"FAIL: Zombie G1 (-10) | Int/OpInc={int_opinc:.2f}")
    if sbc_ocf is not None and sbc_ocf >= 0:
        if sbc_ocf > 0.30:
            pts["sbc_death_ratio"] = -25; notes.append(f'FAIL: SBC Death Ratio G2 (-25) | SBC/OCF={sbc_ocf:.2f}')
        elif sbc_ocf >= 0.10:
            pts["sbc_death_ratio"] = -10; notes.append(f'FAIL: SBC Death Ratio G1 (-10) | SBC/OCF={sbc_ocf:.2f}')
    if ar_g is not None and rev_g is not None:
        if ar_g > (rev_g + 0.10):
            pts["channel_stuffing"] = -25; notes.append("FAIL: Channel Stuffing G2 (-25)")
        elif ar_g > (rev_g + 0.05):
            pts["channel_stuffing"] = -10; notes.append("FAIL: Channel Stuffing G1 (-10)")
    qt = False
    if ni is not None and op_inc is not None and op_inc > 0 and ni > op_inc:
        pts["quality_trap"] = -15; qt = True; notes.append("FAIL: Quality Trap (-15)")

    # Waivers
    if qt and bool(f.get("has_one_off_gain")):
        pts["gain_waiver"] = 5; notes.append("ADJUST: Gain Waiver (+5)")
    if bool(f.get("has_one_off_loss")):
        pts["kitchen_sink"] = 5; notes.append('ADJUST: Kitchen Sink Bonus (+5)')

    total = sum(pts.values())
    return {
        "meta": data_json.get("meta", {}),
        "points": pts, "total": total, "notes": notes,
        "diagnostics": {
            "rev_growth": rev_g, "rpo_growth": rpo_g, "ar_growth": ar_g,
            "gm_bps_change": gm_bps, "fcf_proxy": fcf,
            "ocf_net_income_ratio": ocf_ni, "interest_op_income_ratio": int_opinc,
            "sbc_ocf_ratio": sbc_ocf,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEMINI EXTRACTION (matches Streamlit prompt exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION_PROMPT = """
ROLE: FINANCIAL DATA MINER (STRICT JSON)
OBJECTIVE: Extract GAAP financial data and boolean flags from {doc_type} text.

ABSOLUTE RULES:
- Copy numbers EXACTLY as shown (keep $ , commas, parentheses, etc). DO NOT convert units.
- Extract GAAP only (ignore non-GAAP / adjusted).
- If a value is not present, set it to the STRING "null" (not JSON null).
- Return ONLY valid JSON. No markdown. No commentary.

PERIOD RULES (CRITICAL):
A) Income Statement metrics MUST be for:
   - "Three Months Ended" current quarter (current)
   - "Three Months Ended" prior-year same quarter (prior)

B) Balance Sheet metric (Accounts Receivable):
   - Use period-end amounts for the quarter end date (current)
   - Use prior-year same quarter end date (prior)
   - Line name typically: "Accounts receivable, net" or "Accounts receivable"

C) Cash Flow metrics (OCF + CapEx proxy):
   - First try to find them under a "Three Months Ended" cash flow section IF it exists.
   - If the cash flow statement only shows YTD (common in 10-Q), then extract from the MOST RECENT cash flow period shown
     (e.g., "Nine Months Ended") using the current period column.
   - Do NOT guess quarterly cash flow if only YTD is shown.

WHERE TO LOOK (STRICT):
1) revenue, operating_income, net_income_gaap:
   - Consolidated Statements of Operations (Income Statement)

2) gross_profit (IMPORTANT):
   - If the Income Statement explicitly shows a line item called "Gross profit" (or "Gross profit (loss)"),
     then extract it (current/prior).
   - If the Income Statement DOES NOT show a gross profit line (common for some companies like Amazon),
     set gross_profit.current and gross_profit.prior to the STRING "null".
   - In that case, you MUST still extract cost_of_sales (below). Gross profit will be computed in Python as:
       gross_profit = revenue - cost_of_sales

3) cost_of_sales (REQUIRED WHEN GROSS PROFIT IS NOT SHOWN):
   - Income Statement line like:
     - "Cost of sales"
     - "Cost of goods sold"
     - "Cost of revenue"
     - "Cost of revenues"
   - Extract for Three Months Ended current/prior (same period rules as revenue).

4) accounts_receivable:
   - Consolidated Balance Sheets
   - Line like "Accounts receivable, net" / "Accounts receivable"

5) ocf:
   - Cash Flow Statement line matching ANY of:
     - "Net cash provided by (used for) operating activities"
     - "Net cash provided by (used in) operating activities"
     - "Net cash (used in) provided by operating activities"
   - If this is an 8-K press release and it uses prose, accept:
     - "cash from operations" / "cash provided by operating activities"

6) capex (IMPORTANT: USE INVESTING CASH FLOW AS CAPEX PROXY):
   - Extract the value from the Cash Flow Statement line:
     - "Net cash provided by (used for) investing activities"
     - "Net cash provided by (used in) investing activities"
     - "Net cash (used in) provided by investing activities"
   - Treat this as the CapEx proxy even though it may include other investing items.
   - Use the same period logic as OCF (Three Months if available, else the most recent shown such as Nine Months).

7) interest_expense:
   - "Interest expense" OR "Finance costs"

8) sbc_expense:
   - Income Statement OR Cash Flow non-cash addback line:
     - "Stock-based compensation"
     - "Share-based compensation"

9) rpo:
   - "Remaining performance obligations" OR "RPO" OR "backlog"
   - If not present in the filing text, set to "null".

ONE-OFF KEYWORDS (BOOLEAN FLAGS):
- has_one_off_gain = true if terms like "Gain on sale", "Gain on divestiture", "Divestiture", "Tax benefit" appear.
- has_one_off_loss = true if terms like "Restructuring", "Impairment", "Legal settlement", "Write-down" appear.

REPORTING UNIT:
- Set meta.reporting_unit to the unit stated in the statements (e.g., "Millions", "Thousands").
- If not explicitly stated, use "ones".

PERIOD ENDED:
- meta.period_ended should be the quarter end date string if visible (e.g., "Sep 27, 2025").

STRICT OUTPUT:
Return ONLY a single valid JSON object matching this schema. No markdown. No commentary.

JSON OUTPUT SCHEMA:
{{
  "meta": {{
    "doc_type": "{doc_type}",
    "reporting_unit": "String (e.g., 'Millions')",
    "period_ended": "String"
  }},
  "data": {{
    "revenue": {{ "current": "String", "prior": "String" }},
    "cost_of_sales": {{ "current": "String", "prior": "String" }},
    "gross_profit": {{ "current": "String", "prior": "String" }},
    "operating_income": {{ "current": "String", "prior": "String" }},
    "net_income_gaap": {{ "current": "String", "prior": "String" }},
    "interest_expense": {{ "current": "String" }},
    "ocf": {{ "current": "String" }},
    "capex": {{ "current": "String" }},
    "sbc_expense": {{ "current": "String" }},
    "rpo": {{ "current": "String", "prior": "String" }},
    "accounts_receivable": {{ "current": "String", "prior": "String" }}
  }},
  "flags": {{
    "has_one_off_gain": false,
    "has_one_off_loss": false
  }}
}}

FILING TEXT:
---BEGIN---
{filing_text}
---END---
"""


def _clean_json(raw):
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0).strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)


def _regex_find_first_number_after(text, patterns):
    if not text:
        return None
    t = re.sub(r"\s+", " ", text)
    num = r"(\(?-?\$?\s*[\d,]+(?:\.\d+)?\)?)"
    for pat in patterns:
        m = re.search(pat + r".{0,220}?" + num, t, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if re.search(r"\d", val):
                return val
    return None


def fallback_extract_ocf_capex_ar(filing_text):
    ocf_patterns = [
        r"net cash provided by\s*\(used for\)\s*operating activities",
        r"net cash provided by\s*\(used in\)\s*operating activities",
        r"net cash\s*\(used in\)\s*provided by\s*operating activities",
        r"net cash provided by operating activities",
        r"net cash used in operating activities",
        r"cash from operations",
        r"cash provided by operating activities",
    ]
    capex_patterns = [
        r"net cash provided by\s*\(used for\)\s*investing activities",
        r"net cash provided by\s*\(used in\)\s*investing activities",
        r"net cash\s*\(used in\)\s*provided by\s*investing activities",
        r"net cash provided by investing activities",
        r"net cash used in investing activities",
    ]
    ar_patterns = [
        r"accounts receivable,\s*net",
        r"accounts receivable\s*net",
        r"accounts receivable",
    ]
    return {
        "ocf": _regex_find_first_number_after(filing_text, ocf_patterns),
        "capex": _regex_find_first_number_after(filing_text, capex_patterns),
        "ar": _regex_find_first_number_after(filing_text, ar_patterns),
    }


def extract_with_gemini(filing_text: str, doc_type: str) -> dict:
    MAX_CHARS = 250_000
    filing_text = (filing_text or "")[:MAX_CHARS]

    prompt = EXTRACTION_PROMPT.format(doc_type=doc_type, filing_text=filing_text)
    raw = query_gemini(prompt)

    try:
        data = _clean_json(raw)
    except Exception as e:
        snippet = (raw or "")[:500].replace("\n", "\\n")
        raise ValueError(f"Gemini did not return valid JSON. Error={e}. Snippet={snippet}")

    data.setdefault("meta", {})
    data.setdefault("data", {})
    data.setdefault("flags", {})

    for k, tmpl in [
        ("revenue", {"current": "null", "prior": "null"}),
        ("cost_of_sales", {"current": "null", "prior": "null"}),
        ("gross_profit", {"current": "null", "prior": "null"}),
        ("operating_income", {"current": "null", "prior": "null"}),
        ("net_income_gaap", {"current": "null", "prior": "null"}),
        ("interest_expense", {"current": "null"}),
        ("ocf", {"current": "null"}),
        ("capex", {"current": "null"}),
        ("sbc_expense", {"current": "null"}),
        ("rpo", {"current": "null", "prior": "null"}),
        ("accounts_receivable", {"current": "null", "prior": "null"}),
    ]:
        if k not in data["data"] or not isinstance(data["data"].get(k), dict):
            data["data"][k] = tmpl

    # Regex fallback for missing OCF / CapEx / AR
    def is_missing(x):
        return x is None or str(x).strip().lower() in ("", "n/a", "na", "null")

    ocf_cur = (data["data"].get("ocf", {}) or {}).get("current")
    capex_cur = (data["data"].get("capex", {}) or {}).get("current")
    ar_cur = (data["data"].get("accounts_receivable", {}) or {}).get("current")

    if is_missing(ocf_cur) or is_missing(capex_cur) or is_missing(ar_cur):
        fb = fallback_extract_ocf_capex_ar(filing_text)
        if is_missing(ocf_cur) and fb.get("ocf"):
            data["data"]["ocf"]["current"] = fb["ocf"]
        if is_missing(capex_cur) and fb.get("capex"):
            data["data"]["capex"]["current"] = fb["capex"]
        if is_missing(ar_cur) and fb.get("ar"):
            data["data"]["accounts_receivable"]["current"] = fb["ar"]

    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAIR FINDING (8K <-> 10Q)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_next_10q(cik, anchor_date, max_days=120):
    data = get_submissions(cik)
    if not data:
        return None
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    reports = recent.get("reportDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    n = min(len(forms), len(dates), len(accessions), len(docs))
    latest = anchor_date + timedelta(days=max_days)
    for i in range(n):
        if str(forms[i]).strip() != "10-Q":
            continue
        fd = _parse_date(dates[i])
        if fd and anchor_date <= fd <= latest:
            return {
                "form": "10-Q",
                "accession_number": accessions[i],
                "primary_document": docs[i],
                "filing_date": dates[i],
                "report_period": reports[i] if i < len(reports) else None,
                "acceptanceDateTime": acceptances[i] if i < len(acceptances) else None,
            }
    return None


def _find_prev_8k(cik, anchor_date, lookback=120):
    data = get_submissions(cik)
    if not data:
        return None
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    reports = recent.get("reportDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    n = min(len(forms), len(dates), len(accessions), len(docs))
    earliest = anchor_date - timedelta(days=lookback)
    checked = 0
    for i in range(n):
        if str(forms[i]).strip() != "8-K":
            continue
        fd = _parse_date(dates[i])
        if not fd or fd > anchor_date or fd < earliest:
            continue
        checked += 1
        if checked > 15:
            break
        try:
            html = fetch_filing_html(cik, accessions[i], docs[i])
            if filing_contains_item_202(html):
                return {
                    "form": "8-K",
                    "accession_number": accessions[i],
                    "primary_document": docs[i],
                    "filing_date": dates[i],
                    "report_period": reports[i] if i < len(reports) else None,
                    "acceptanceDateTime": acceptances[i] if i < len(acceptances) else None,
                }
        except Exception:
            continue
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.after_request
def after_request(resp):
    return _cors(resp)


@app.route("/api/metadata", methods=["GET", "OPTIONS"])
def metadata():
    if request.method == "OPTIONS":
        return _cors(Response("", 204))
    try:
        meta = load_ticker_metadata()
        return jsonify({"ok": True, "count": len(meta), "data": meta})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/scan", methods=["POST", "OPTIONS"])
def scan():
    if request.method == "OPTIONS":
        return _cors(Response("", 204))
    try:
        body = request.get_json(force=True)
        tickers = body.get("tickers", [])
        days = int(body.get("days", 30))
        meta = load_ticker_metadata()
        results = []
        for t in tickers[:15]:
            t = t.upper().strip()
            info = meta.get(t)
            if not info:
                continue
            cik = info["cik"]
            filings = get_qualifying_filings(cik, days=days)
            if filings:
                results.append({
                    "ticker": t,
                    "company": info["name"],
                    "cik": cik,
                    "filings": filings,
                })
            time.sleep(0.11)
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/filing", methods=["GET", "OPTIONS"])
def filing():
    if request.method == "OPTIONS":
        return _cors(Response("", 204))
    try:
        cik = request.args.get("cik", "")
        acc = request.args.get("accession", "")
        doc = request.args.get("doc", "")
        form = request.args.get("form", "")
        if form == "8-K":
            html = fetch_effective_8k_html(cik, acc, doc)
        else:
            html = fetch_filing_html(cik, acc, doc)
        return jsonify({"ok": True, "html": html})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return _cors(Response("", 204))
    try:
        body = request.get_json(force=True)
        cik = body.get("cik", "")
        ticker = body.get("ticker", "")
        company = body.get("company", "")

        eight_k = body.get("eight_k")
        ten_q = body.get("ten_q")

        out = {"eight_k": None, "ten_q": None}

        if eight_k:
            html = fetch_effective_8k_html(cik, eight_k["accession_number"], eight_k["primary_document"])
            text = html_to_text(html)
            data_json = extract_with_gemini(text, "8K")
            scorecard = compute_scorecard(data_json)
            out["eight_k"] = {"json": data_json, "scorecard": scorecard}

        if ten_q:
            html = fetch_filing_html(cik, ten_q["accession_number"], ten_q["primary_document"])
            text = html_to_text(html)
            data_json = extract_with_gemini(text, "10Q")
            scorecard = compute_scorecard(data_json)
            out["ten_q"] = {"json": data_json, "scorecard": scorecard}

        return jsonify({"ok": True, "ticker": ticker, "company": company, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/pair", methods=["GET", "OPTIONS"])
def pair():
    if request.method == "OPTIONS":
        return _cors(Response("", 204))
    try:
        cik = request.args.get("cik", "")
        form = request.args.get("form", "")
        filing_date = request.args.get("filing_date", "")
        fd = _parse_date(filing_date)
        if not fd:
            return jsonify({"ok": False, "error": "Invalid filing_date"}), 400

        if form == "8-K":
            pair_filing = _find_next_10q(cik, fd)
        elif form == "10-Q":
            pair_filing = _find_prev_8k(cik, fd)
        else:
            pair_filing = None

        return jsonify({"ok": True, "pair": pair_filing})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})
