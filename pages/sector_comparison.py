import io
import os
from datetime import date, datetime, timedelta

import anthropic
import markdown as md
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from xhtml2pdf import pisa

from utils import ETF_TO_GICS, MODELS, estimate_cost

load_dotenv()

st.set_page_config(page_title="Sector Comparison", layout="wide")

SECTOR_ETFS = list(ETF_TO_GICS.keys())  # 11 ETFs
GICS_TO_ETF = {v: k for k, v in ETF_TO_GICS.items()}

GROUPS = {
    "Sensitive":  ["XLC", "XLE", "XLI", "XLK"],
    "Cyclical":   ["XLB", "XLF", "XLRE", "XLY"],
    "Defensive":  ["XLP", "XLU", "XLV"],
}


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_data(start_date: date, end_date: date):
    tickers     = SECTOR_ETFS + ["SPY"]
    fetch_start = (start_date - timedelta(days=10)).isoformat()
    fetch_end   = (end_date   + timedelta(days=1)).isoformat()

    ohlcv = yf.download(tickers, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    close = ohlcv["Close"]
    high  = ohlcv["High"]
    low   = ohlcv["Low"]

    base     = close[close.index.date <= start_date].iloc[-1]
    period_c = close[(close.index.date >= start_date) & (close.index.date <= end_date)]
    latest   = period_c.iloc[-1]
    ret      = ((latest - base) / base * 100).round(2)

    # 52-week stats
    w52 = close[close.index.date <= end_date].tail(252)
    hi52 = w52[SECTOR_ETFS].max()
    lo52 = w52[SECTOR_ETFS].min()

    # Parkinson volatility (annualised)
    w_h   = high[SECTOR_ETFS][high.index.date <= end_date].tail(252)
    w_l   = low[SECTOR_ETFS][low.index.date   <= end_date].tail(252)
    lhl   = np.log(w_h / w_l)
    park  = (np.sqrt((lhl**2).mean() / (4 * np.log(2)) * 252) * 100).round(2)

    # Beta vs SPY
    rets    = close.pct_change().dropna()
    spy_var = rets["SPY"].var()
    betas   = {t: round(rets[t].cov(rets["SPY"]) / spy_var, 2) for t in SECTOR_ETFS if t in rets.columns}

    rows = []
    for t in SECTOR_ETFS:
        cur  = float(latest.get(t, np.nan))
        h52  = float(hi52.get(t, np.nan))
        l52  = float(lo52.get(t, np.nan))
        grp  = next((g for g, ts in GROUPS.items() if t in ts), "")
        rows.append({
            "ETF":           t,
            "Sector":        ETF_TO_GICS[t],
            "Group":         grp,
            "Return (%)":    round(float(ret.get(t, np.nan)), 2),
            "Price":         round(cur, 2),
            "52W High":      round(h52, 2),
            "52W Low":       round(l52, 2),
            "% from High":   round((cur - h52) / h52 * 100, 2) if h52 else None,
            "% from Low":    round((cur - l52) / l52 * 100, 2) if l52 else None,
            "Beta":          betas.get(t),
            "Parkinson Vol": round(float(park.get(t, np.nan)), 2),
        })

    df = pd.DataFrame(rows).sort_values("Return (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Daily return series for SPY benchmark line
    series = ((period_c[SECTOR_ETFS + ["SPY"]] - base[SECTOR_ETFS + ["SPY"]]) / base[SECTOR_ETFS + ["SPY"]] * 100).round(4)
    series.index = series.index.date

    return df, series


def build_comparison_prompt(df, start_date, end_date, preset):
    period = f"{start_date.strftime('%m-%d-%Y')} -> {end_date.strftime('%m-%d-%Y')}"
    table  = df.to_string(index=False)
    return f"""You are a senior equity strategist with expertise across all 11 S&P 500 GICS sectors.

## Analysis Parameters
- Period: {period} ({preset})
- Analysis Date: {date.today().strftime('%m-%d-%Y')}
- Data: All 11 S&P 500 sector ETFs ranked by period return

## Sector Performance Data
{table}

## Your Task

Provide a comprehensive cross-sector analysis structured as follows:

### 1. Market Regime Assessment
In 2-3 sentences, characterize the current market environment based on which sectors are leading/lagging. Is this a risk-on or risk-off regime? What macro story does the sector rotation tell?

### 2. Sector Rankings & Outlook
For each sector (in order from most to least attractive), provide:
- **Rank # — Sector (ETF): Bullish / Neutral / Bearish**
- One sentence: the key reason for your stance (reference the data)

### 3. Top 3 Sector Overweights
For each overweight:
- **ETF — Sector**
- Investment thesis (2-3 sentences): why this sector now, what catalysts, what risks

### 4. Top 2 Sector Underweights / Avoids
For each:
- **ETF — Sector**
- Why to underweight: valuation, momentum, fundamental headwinds

### 5. Key Rotation Signals
Identify 2-3 specific rotation trades (e.g., "rotate from X into Y") with the reasoning grounded in the data.

### 6. Portfolio Positioning Summary
A 3-bullet summary of the overall recommended sector allocation tilt.

Be specific and reference actual return/volatility/beta numbers from the data.
"""


# ── PDF export ────────────────────────────────────────────────────────────────

def generate_pdf(df, start_date, end_date, preset, model_id, analysis_text):
    # Data table as HTML
    def fmt(v, col):
        if pd.isna(v):
            return "—"
        if col in ("Return (%)", "% from High", "% from Low"):
            color = "#1a9850" if float(v) >= 0 else "#d73027"
            return f'<span style="color:{color};font-weight:bold">{float(v):+.2f}%</span>'
        if col in ("Price", "52W High", "52W Low"):
            return f"${float(v):.2f}"
        if col == "Parkinson Vol":
            return f"{float(v):.2f}%"
        if col == "Beta":
            return f"{float(v):.2f}"
        return str(v)

    header_cells = "".join(f"<th>{c}</th>" for c in df.columns)
    data_rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{fmt(row[c], c)}</td>" for c in df.columns)
        data_rows += f"<tr>{cells}</tr>"

    table_html = f"""
    <table>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{data_rows}</tbody>
    </table>"""

    analysis_html = md.markdown(analysis_text, extensions=["tables", "nl2br", "fenced_code"])
    period        = f"{start_date.strftime('%m-%d-%Y')} → {end_date.strftime('%m-%d-%Y')}"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ size: A4 landscape; margin: 15mm 12mm 15mm 12mm; }}
  body   {{ font-family: Helvetica, Arial, sans-serif; font-size: 9pt; color: #222; line-height: 1.4; }}
  .report-header  {{ margin-bottom: 10px; border-bottom: 2px solid #1e3a5f; padding-bottom: 6px; }}
  .report-title   {{ font-size: 16pt; font-weight: bold; color: #1e3a5f; margin: 0 0 3px 0; }}
  .report-meta    {{ font-size: 8pt; color: #666; }}
  h1 {{ font-size: 13pt; color: #1e3a5f; border-bottom: 1px solid #c0c8d8; padding-bottom: 3px; margin-top: 12px; }}
  h2 {{ font-size: 11pt; color: #fff; background-color: #1e3a5f; padding: 3px 7px; margin-top: 12px; }}
  h3 {{ font-size: 10pt; color: #1e3a5f; margin-top: 8px; border-left: 3px solid #1e3a5f; padding-left: 5px; }}
  h4 {{ font-size: 9pt; color: #333; margin-top: 6px; }}
  table  {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 7.5pt; }}
  th     {{ background-color: #1e3a5f; color: #fff; padding: 4px 6px; text-align: left; font-weight: bold; white-space: nowrap; }}
  td     {{ padding: 3px 6px; border-bottom: 1px solid #ddd; vertical-align: top; white-space: nowrap; }}
  tr:nth-child(even) td {{ background-color: #f2f5fa; }}
  ul, ol {{ margin: 3px 0 6px 0; padding-left: 16px; }}
  li     {{ margin: 2px 0; }}
  strong {{ color: #1e3a5f; }}
  hr     {{ border: none; border-top: 1px solid #ccc; margin: 8px 0; }}
  p      {{ margin: 4px 0; }}
</style>
</head>
<body>
<div class="report-header">
  <div class="report-title">S&P 500 Sector Comparison &mdash; Cross-Sector Analysis</div>
  <div class="report-meta">
    Period: <strong>{period}</strong> ({preset}) &nbsp;|&nbsp;
    Model: <strong>{model_id}</strong> &nbsp;|&nbsp;
    Generated: <strong>{date.today().strftime('%m-%d-%Y')}</strong>
  </div>
</div>

<h2>Sector Performance Data</h2>
{table_html}

<h2>AI Cross-Sector Analysis</h2>
{analysis_html}

<hr>
<p style="font-size:6.5pt;color:#999;text-align:center;">
  Generated by Sector Comparison &nbsp;|&nbsp; {date.today().strftime('%m-%d-%Y')} &nbsp;|&nbsp; For informational purposes only. Not financial advice.
</p>
</body>
</html>"""

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer, encoding="utf-8")
    return pdf_buffer.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    _env_key = os.getenv("ANTHROPIC_API_KEY", "")
    if _env_key:
        api_key = _env_key
    else:
        api_key = st.text_input("Anthropic API Key", type="password")

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model       = MODELS[model_label]

    st.divider()
    st.header("Period")

    today   = date.today()
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
    run = st.button("Run Comparison", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Sector Comparison")
st.caption("Cross-sector performance, rotation signals, and AI-powered ranking across all 11 S&P 500 sectors.")

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar.")

period_label = f"{start_date.strftime('%m-%d-%Y')} → {end_date.strftime('%m-%d-%Y')}"

# Always show the data table, regardless of whether analysis has been run
try:
    with st.spinner("Loading sector data..."):
        df, series = fetch_sector_data(start_date, end_date)

    # ── Heatmap-style return bar ───────────────────────────────────────────────
    GROUP_COLORS = {"Sensitive": "#f59e0b", "Cyclical": "#3b82f6", "Defensive": "#22c55e"}

    fig = px.bar(
        df,
        x="Sector",
        y="Return (%)",
        color="Group",
        color_discrete_map=GROUP_COLORS,
        text=df["Return (%)"].apply(lambda v: f"{v:+.2f}%"),
        title=f"Sector Returns — {period_label}",
        height=460,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_ticksuffix="%", xaxis_tickangle=-30, legend_title="Group")
    st.plotly_chart(fig, width='stretch')

    # ── Metrics row ───────────────────────────────────────────────────────────
    best  = df.iloc[0]
    worst = df.iloc[-1]
    cols  = st.columns(4)
    cols[0].metric("Best Sector",    f"{best['ETF']}",  f"{best['Return (%)']:+.2f}%")
    cols[1].metric("Worst Sector",   f"{worst['ETF']}", f"{worst['Return (%)']:+.2f}%")
    cols[2].metric("Spread (Best−Worst)", f"{best['Return (%)'] - worst['Return (%)']:.2f}%")
    spy_ret = float(series["SPY"].iloc[-1]) if "SPY" in series.columns else 0
    cols[3].metric("SPY Benchmark",  f"{spy_ret:+.2f}%")

    # ── Data table ────────────────────────────────────────────────────────────
    st.subheader(f"All Sectors — {period_label}")

    def color_ret(val):
        color = "#1a9850" if val > 0 else "#d73027" if val < 0 else "gray"
        return f"color: {color}; font-weight: bold"

    styled = (
        df.style
        .applymap(color_ret, subset=["Return (%)", "% from High", "% from Low"])
        .format({
            "Return (%)":    "{:+.2f}%",
            "% from High":   "{:+.2f}%",
            "% from Low":    "{:+.2f}%",
            "Price":         "${:.2f}",
            "52W High":      "${:.2f}",
            "52W Low":       "${:.2f}",
            "Beta":          "{:.2f}",
            "Parkinson Vol": "{:.2f}%",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=450)

    # ── Line chart ────────────────────────────────────────────────────────────
    st.subheader("Performance Over Time")
    long_df = (
        series[SECTOR_ETFS + ["SPY"]]
        .reset_index().rename(columns={"index": "Date"})
        .melt(id_vars="Date", var_name="ETF", value_name="Return (%)")
    )
    line_fig = px.line(long_df, x="Date", y="Return (%)", color="ETF",
                       title=f"Cumulative Return (%) — {period_label}", height=480)
    line_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    line_fig.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
    line_fig.update_traces(selector={"name": "SPY"}, line=dict(color="black", width=2.5, dash="dot"))
    st.plotly_chart(line_fig, width='stretch')

    # ── AI Analysis ───────────────────────────────────────────────────────────
    if run:
        if not api_key:
            st.error("API key required to run analysis.")
            st.stop()

        st.divider()
        st.subheader("AI Cross-Sector Analysis")
        model_name = model_label.split("(")[0].strip()
        st.caption(f"Powered by {model_name}")

        prompt    = build_comparison_prompt(df, start_date, end_date, preset)
        client    = anthropic.Anthropic(api_key=api_key)
        full_text = ""

        try:
            output = st.empty()
            with client.messages.stream(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    output.markdown(full_text + "▌")
            output.markdown(full_text)

            usage    = stream.get_final_message().usage
            cost_str = estimate_cost(model, usage.input_tokens, usage.output_tokens)
            st.caption(
                f"Tokens — input: {usage.input_tokens:,} · output: {usage.output_tokens:,} · "
                f"est. cost: {cost_str} · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            pdf_filename = f"sector_comparison_{preset}_{date.today().strftime('%Y%m%d')}.pdf"
            pdf_bytes    = generate_pdf(df, start_date, end_date, preset, model_label.split("(")[0].strip(), full_text)
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
        st.info("Click **Run Comparison** in the sidebar to generate AI-powered cross-sector analysis.")

except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("Check your internet connection and try refreshing.")
