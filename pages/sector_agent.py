import streamlit as st
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import markdown as md
from xhtml2pdf import pisa
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import datetime
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Sector Investment Agent", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

ETF_TO_GICS = {
    "XLC":  "Communication Services",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLK":  "Information Technology",
    "XLB":  "Materials",
    "XLF":  "Financials",
    "XLRE": "Real Estate",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLV":  "Health Care",
}

MODELS = {
    "Claude Sonnet 4.6 (Recommended)": "claude-sonnet-4-6",
    "Claude Opus 4.6 (Most Capable)":  "claude-opus-4-6",
    "Claude Haiku 4.5 (Fastest)":      "claude-haiku-4-5-20251001",
}

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_constituents():
    gics_to_etf = {v: k for k, v in ETF_TO_GICS.items()}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    df = pd.read_html(resp.text)[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["Ticker", "Name", "GICS Sector", "Sub-Industry"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["ETF"]    = df["GICS Sector"].map(gics_to_etf)
    return df.dropna(subset=["ETF"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_data(tickers_tuple, start_date: date, end_date: date):
    tickers     = list(tickers_tuple)
    dl          = tickers + ([] if "SPY" in tickers else ["SPY"])
    fetch_start = (start_date - timedelta(days=10)).isoformat()
    fetch_end   = (end_date   + timedelta(days=1)).isoformat()

    ohlcv = yf.download(dl, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        high  = high.to_frame(tickers[0])
        low   = low.to_frame(tickers[0])

    base     = close[close.index.date <= start_date].iloc[-1]
    period_c = close[(close.index.date >= start_date) & (close.index.date <= end_date)]
    latest   = period_c.iloc[-1]
    ret      = ((latest - base) / base * 100).round(2)

    wc  = close[tickers][close.index.date <= end_date].tail(252)
    wh  = high[tickers][high.index.date   <= end_date].tail(252)
    wl  = low[tickers][low.index.date     <= end_date].tail(252)
    hi52 = wc.max()
    lo52 = wc.min()

    rets    = close.pct_change().dropna()
    spy_var = rets["SPY"].var() if "SPY" in rets.columns else None
    betas   = {}
    if spy_var and spy_var > 0:
        for t in tickers:
            if t in rets.columns:
                betas[t] = round(rets[t].cov(rets["SPY"]) / spy_var, 2)

    park = {}
    for t in tickers:
        if t in wh.columns:
            lhl = np.log(wh[t] / wl[t])
            park[t] = round(np.sqrt((lhl**2).mean() / (4 * np.log(2)) * 252) * 100, 2)

    rows = []
    for t in tickers:
        cur = float(latest.get(t, np.nan))
        h52 = float(hi52.get(t, np.nan))
        l52 = float(lo52.get(t, np.nan))
        rows.append({
            "Ticker":          t,
            "Price":           round(cur, 2),
            "Return (%)":      round(float(ret.get(t, np.nan)), 2),
            "52W High":        round(h52, 2),
            "52W Low":         round(l52, 2),
            "% from High":     round((cur - h52) / h52 * 100, 2) if h52 else None,
            "% from Low":      round((cur - l52) / l52 * 100, 2) if l52 else None,
            "Beta":            betas.get(t),
            "Parkinson Vol":   park.get(t),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamentals(tickers_tuple):
    def fetch_one(ticker):
        try:
            info = yf.Ticker(ticker).info
            mc   = info.get("marketCap")
            pe   = info.get("trailingPE")
            fpe  = info.get("forwardPE")
            dy   = info.get("dividendYield")
            eg   = info.get("earningsGrowth")
            return {
                "Ticker":       ticker,
                "Mkt Cap ($B)": round(mc / 1e9, 1) if mc  else None,
                "P/E":          round(pe, 1)        if pe  else None,
                "Fwd P/E":      round(fpe, 1)       if fpe else None,
                "Div Yield (%)":round(dy, 2)        if dy  else None,
                "EPS Growth (%)":round(eg * 100, 1) if eg  else None,
            }
        except Exception:
            return {"Ticker": ticker}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_one, list(tickers_tuple)))
    return pd.DataFrame(results)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(sector_name, etf, start_date, end_date, data, sp500):
    period = f"{start_date.strftime('%m-%d-%Y')} -> {end_date.strftime('%m-%d-%Y')}"

    # Sub-industry summary
    grp = (
        data.groupby("Sub-Industry")
        .agg(
            Stocks       =("Ticker",      "count"),
            Avg_Return   =("Return (%)",  "mean"),
            Pct_Positive =("Return (%)",  lambda x: (x > 0).mean() * 100),
        )
        .round(2)
        .sort_values("Avg_Return", ascending=False)
        .reset_index()
    )
    sub_industry_text = grp.to_string(index=False)

    # Stock table
    cols = ["Ticker", "Name", "Sub-Industry", "Mkt Cap ($B)", "Return (%)",
            "% from High", "% from Low", "Beta", "Parkinson Vol",
            "P/E", "Fwd P/E", "Div Yield (%)", "EPS Growth (%)"]
    cols = [c for c in cols if c in data.columns]
    stock_text = data[cols].sort_values("Return (%)", ascending=False).to_string(index=False)

    prompt = f"""You are an expert investment analyst with deep expertise in the {sector_name} sector.

CRITICAL RULE: Your analysis must be COMPLETELY ISOLATED to the {sector_name} sector. Do not compare with, reference, or consider any other sectors whatsoever. Treat this sector as if it is the only sector that exists.

## Analysis Parameters
- Sector: {sector_name} (ETF: {etf})
- Analysis Period: {period}
- Universe: S&P 500 {sector_name} constituents ({len(data)} stocks)
- Analysis Date: {date.today().strftime('%m-%d-%Y')}

## Sub-Industry Performance
{sub_industry_text}

## Stock Fundamentals & Technicals
{stock_text}

## Your Task

Please provide a thorough investment analysis structured as follows:

### 1. Macro Environment
Assess the current macroeconomic environment as it specifically and uniquely affects the {sector_name} sector. Consider interest rates, inflation, regulatory landscape, technological trends, and any sector-specific macro drivers. Be specific to this sector only.

### 2. Sub-Industry Outlook
For each sub-industry, provide a brief outlook (Bullish / Neutral / Bearish) with one sentence of reasoning based on the data and macro context.

### 3. Top Investment Picks (3-5 stocks)
For each pick:
- **Ticker — Company Name**
- Investment thesis (2-3 sentences)
- Key supporting data points (return, valuation, momentum)
- Risk factors

### 4. Stocks to Avoid (2-3 stocks)
For each:
- **Ticker — Company Name**
- Why to avoid (valuation concern, weak momentum, fundamental deterioration, etc.)

### 5. Sector Verdict
- Overall stance: **Bullish / Neutral / Bearish**
- Conviction level: High / Medium / Low
- 2-3 sentence summary of the investment case for this sector right now

Be specific, data-driven, and actionable. Reference actual numbers from the data provided.
"""
    return prompt


# ── PDF export ───────────────────────────────────────────────────────────────

def generate_pdf(sector_name, etf, start_date, end_date, analysis_text, num_stocks):
    # Convert markdown to HTML (tables extension handles | tables natively)
    body_html = md.markdown(
        analysis_text,
        extensions=["tables", "nl2br", "fenced_code"],
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
  }}

  body   {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #222; line-height: 1.5; }}

  /* Header block */
  .report-header  {{ margin-bottom: 12px; border-bottom: 2px solid #1e3a5f; padding-bottom: 8px; }}
  .report-title   {{ font-size: 18pt; font-weight: bold; color: #1e3a5f; margin: 0 0 4px 0; }}
  .report-meta    {{ font-size: 9pt; color: #666; }}

  /* Headings */
  h1 {{ font-size: 15pt; color: #1e3a5f; border-bottom: 1px solid #c0c8d8; padding-bottom: 3px; margin-top: 14px; }}
  h2 {{ font-size: 13pt; color: #fff; background-color: #1e3a5f; padding: 4px 8px; margin-top: 14px; }}
  h3 {{ font-size: 11pt; color: #1e3a5f; margin-top: 10px; border-left: 3px solid #1e3a5f; padding-left: 6px; }}
  h4 {{ font-size: 10pt; color: #333; margin-top: 8px; }}

  /* Tables */
  table  {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8.5pt; }}
  th     {{ background-color: #1e3a5f; color: #fff; padding: 5px 7px; text-align: left; font-weight: bold; }}
  td     {{ padding: 4px 7px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  tr:nth-child(even) td {{ background-color: #f2f5fa; }}

  /* Lists */
  ul, ol {{ margin: 4px 0 8px 0; padding-left: 18px; }}
  li     {{ margin: 3px 0; }}

  /* Misc */
  strong {{ color: #1e3a5f; }}
  hr     {{ border: none; border-top: 1px solid #ccc; margin: 10px 0; }}
  p      {{ margin: 5px 0; }}
  code   {{ background: #f0f0f0; padding: 1px 4px; font-size: 9pt; }}
</style>
</head>
<body>

<div class="report-header">
  <div class="report-title">{sector_name} Sector &mdash; Investment Analysis</div>
  <div class="report-meta">
    ETF: <strong>{etf}</strong> &nbsp;|&nbsp;
    Period: <strong>{start_date.strftime('%m-%d-%Y')} &rarr; {end_date.strftime('%m-%d-%Y')}</strong> &nbsp;|&nbsp;
    Universe: <strong>{sector_name} ({num_stocks} Stocks)</strong> &nbsp;|&nbsp;
    Generated: <strong>{date.today().strftime('%m-%d-%Y')}</strong>
  </div>
</div>

{body_html}

<hr>
<p style="font-size:7pt; color:#999; text-align:center;">
  Generated by Sector Investment Agent &nbsp;|&nbsp; {date.today().strftime('%m-%d-%Y')} &nbsp;|&nbsp; For informational purposes only. Not financial advice.
</p>

</body>
</html>"""

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer, encoding="utf-8")
    return pdf_buffer.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Agent Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
    )

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model       = MODELS[model_label]

    st.divider()
    st.header("Period")

    today         = date.today()
    PRESETS = {
        "YTD": date(today.year, 1, 1),
        "1Y":  today - timedelta(days=365),
        "2Y":  today - timedelta(days=365*2),
        "3Y":  today - timedelta(days=365*3),
    }
    preset     = st.radio("Quick Select", list(PRESETS.keys()), horizontal=True, index=0)
    start_date = st.date_input("Start Date", value=PRESETS[preset], max_value=today)
    end_date   = st.date_input("End Date",   value=today, min_value=start_date)

    st.divider()
    run = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Sector Investment Agent")
st.caption(f"Powered by {model_label.split('(')[0].strip()} · S&P 500 constituents · Fundamental + Technical analysis")

sector_options = list(ETF_TO_GICS.values())
sector_name    = st.radio("Select Sector", sector_options, horizontal=True, index=0)
etf            = {v: k for k, v in ETF_TO_GICS.items()}[sector_name]

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar.")
    st.stop()

if run:
    # Load data
    with st.spinner("Loading S&P 500 constituents..."):
        sp500 = get_sp500_constituents()

    sector_stocks = sp500[sp500["ETF"] == etf]["Ticker"].tolist()
    name_map      = sp500.set_index("Ticker")[["Name", "Sub-Industry"]].to_dict("index")

    if not sector_stocks:
        st.error("No constituents found for this sector.")
        st.stop()

    st.info(f"**{sector_name}** ({etf}) · {len(sector_stocks)} constituents · {start_date.strftime('%m-%d-%Y')} -> {end_date.strftime('%m-%d-%Y')}")

    with st.spinner(f"Fetching price data for {len(sector_stocks)} stocks..."):
        tickers_t  = tuple(sorted(sector_stocks))
        price_data = get_price_data(tickers_t, start_date, end_date)

    with st.spinner("Fetching fundamental data (cached daily)..."):
        fund_data = get_fundamentals(tickers_t)

    data = price_data.merge(fund_data, on="Ticker", how="left")
    data["Name"]         = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Name", t))
    data["Sub-Industry"] = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Sub-Industry", ""))
    data = data.sort_values("Return (%)", ascending=False).reset_index(drop=True)

    st.success(f"Data loaded. Running {model_label.split('(')[0].strip()} analysis...")
    st.divider()

    # Stream analysis
    prompt = build_prompt(sector_name, etf, start_date, end_date, data, sp500)
    client = anthropic.Anthropic(api_key=api_key)

    output = st.empty()
    full_text = ""

    try:
        with client.messages.stream(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                output.markdown(full_text + "▌")

        output.markdown(full_text)

        # Token usage
        usage = stream.get_final_message().usage
        st.divider()
        st.caption(f"Tokens used — input: {usage.input_tokens:,} · output: {usage.output_tokens:,} · {datetime.now()}")

        # PDF export
        pdf_filename = f"{sector_name.replace(' ', '_')}_{etf}_{preset}_{date.today().strftime('%Y%m%d')}.pdf"
        pdf_bytes    = generate_pdf(sector_name, etf, start_date, end_date, full_text, len(data))
        st.download_button(
            label="Export to PDF",
            data=bytes(pdf_bytes),
            file_name=pdf_filename,
            mime="application/pdf",
            type="primary",
        )

    except anthropic.AuthenticationError:
        st.error("Invalid API key.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.markdown("""
    ### How to use
    1. Select a **sector** and **date range** in the sidebar
    2. Click **Run Analysis**
    3. The agent will load all S&P 500 constituents for that sector, fetch fundamental and technical data, and generate a thorough investment analysis

    ### What the agent analyzes
    - **Macro environment** specific to the sector
    - **Sub-industry outlook** based on performance data
    - **Top picks** with data-driven investment thesis
    - **Stocks to avoid** with reasoning
    - **Overall sector verdict** with conviction level
    """)
