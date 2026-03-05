import base64

import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests

st.set_page_config(page_title="S&P 500 Sector ETF YTD Returns", layout="wide")

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

GROUPS = {
    "Sensitive":  ["XLC", "XLE", "XLI", "XLK"],
    "Cyclical":   ["XLB", "XLF", "XLRE", "XLY"],
    "Defensive":  ["XLP", "XLU", "XLV"],
}
GROUP_COLORS = {
    "Sensitive": "#f59e0b",   # amber
    "Cyclical":  "#3b82f6",   # blue
    "Defensive": "#22c55e",   # green
    "SPY":       "#6b7280",   # gray
}

TICKERS = {
    "SPY": "S&P 500 (SPY)",
    "RSP": "Equal Weight S&P 500 (RSP)",
    "XLC": "Communication Services (XLC)",
    "XLE": "Energy (XLE)",
    "XLI": "Industrials (XLI)",
    "XLK": "Technology (XLK)",
    "XLB": "Materials (XLB)",
    "XLF": "Financials (XLF)",
    "XLRE": "Real Estate (XLRE)",
    "XLY": "Consumer Discretionary (XLY)",
    "XLP": "Consumer Staples (XLP)",
    "XLU": "Utilities (XLU)",
    "XLV": "Health Care (XLV)",
}

# ── SVG price-bar helpers ──────────────────────────────────────────────────────

def make_price_bar_svg(ytd_low, year_start, current, ytd_high, width=540, height=26, inline=False):
    """
    Horizontal range bar:
      Above bar: year_start (gray) and current (colored, bold) — collision-aware
      Below bar: YTD low (left) and YTD high (right)
      On bar:    colored fill from year_start→current, diamond at year_start, dot at current
    """
    pad_l, pad_r = 40, 40
    bar_y, bar_h = 12, 4
    usable = width - pad_l - pad_r

    lo  = min(ytd_low, year_start)
    hi  = max(ytd_high, year_start)
    rng = max(hi - lo, 0.01)

    def xp(price):
        return pad_l + (price - lo) / rng * usable

    x_lo    = xp(lo)
    x_hi    = xp(hi)
    x_start = xp(year_start)
    x_curr  = xp(current)

    color  = "#1a9850" if current >= year_start else "#d73027"
    fill_x = min(x_start, x_curr)
    fill_w = max(abs(x_curr - x_start), 1)
    bar_mid = bar_y + bar_h / 2

    # ── above-bar labels (year_start gray, current colored) ────────────────────
    above_y  = bar_y - 5        # text baseline row above the bar
    min_gap  = 46               # min px between the two label centres
    raw_sx, raw_cx = x_start, x_curr
    if abs(raw_cx - raw_sx) < min_gap:
        mid = (raw_sx + raw_cx) / 2
        half = min_gap / 2
        raw_sx, raw_cx = (mid - half, mid + half) if x_start <= x_curr else (mid + half, mid - half)
    start_lbl_x = max(pad_l + 12, min(width - pad_r - 12, raw_sx))
    curr_lbl_x  = max(pad_l + 12, min(width - pad_r - 12, raw_cx))

    # ── diamond marker (year_start, just above bar) ────────────────────────────
    dm = 2
    dx, dy = x_start, bar_y - 2
    diamond = f"{dx:.1f},{dy-dm:.1f} {dx+dm:.1f},{dy:.1f} {dx:.1f},{dy+dm:.1f} {dx-dm:.1f},{dy:.1f}"

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;display:block">'
        if inline else
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    )
    svg = f"""{svg_open}
  <!-- Track -->
  <rect x="{x_lo:.1f}" y="{bar_y}" width="{x_hi-x_lo:.1f}" height="{bar_h}" rx="2" fill="#d9d9d9"/>
  <!-- Colored fill (year-start → current) -->
  <rect x="{fill_x:.1f}" y="{bar_y}" width="{fill_w:.1f}" height="{bar_h}" fill="{color}" opacity="0.85" rx="1"/>
  <!-- Year-start dashed line -->
  <line x1="{x_start:.1f}" y1="{bar_y-4:.1f}" x2="{x_start:.1f}" y2="{bar_y+bar_h:.1f}"
        stroke="#555" stroke-width="1" stroke-dasharray="2,1.5"/>
  <!-- Year-start diamond -->
  <polygon points="{diamond}" fill="#555"/>
  <!-- Low end-cap -->
  <line x1="{x_lo:.1f}" y1="{bar_y-1}" x2="{x_lo:.1f}" y2="{bar_y+bar_h+1}" stroke="#999" stroke-width="1"/>
  <!-- High end-cap -->
  <line x1="{x_hi:.1f}" y1="{bar_y-1}" x2="{x_hi:.1f}" y2="{bar_y+bar_h+1}" stroke="#999" stroke-width="1"/>
  <!-- Current price dot -->
  <circle cx="{x_curr:.1f}" cy="{bar_mid:.1f}" r="3" fill="{color}" stroke="white" stroke-width="1"/>
  <!-- Year-start label (above bar, gray) -->
  <text x="{start_lbl_x:.1f}" y="{above_y:.1f}" font-size="7.5" fill="#555"
        font-family="sans-serif" text-anchor="middle">${year_start:.2f}</text>
  <!-- Current price label (above bar, bold, colored) -->
  <text x="{curr_lbl_x:.1f}" y="{above_y:.1f}" font-size="7.5" fill="{color}"
        font-weight="bold" font-family="sans-serif" text-anchor="middle">${current:.2f}</text>
  <!-- YTD Low label (below bar, left) -->
  <text x="{x_lo:.1f}" y="{bar_y+bar_h+9:.1f}" font-size="7.5" fill="#888"
        font-family="sans-serif" text-anchor="start">${ytd_low:.2f}</text>
  <!-- YTD High label (below bar, right) -->
  <text x="{x_hi:.1f}" y="{bar_y+bar_h+9:.1f}" font-size="7.5" fill="#888"
        font-family="sans-serif" text-anchor="end">${ytd_high:.2f}</text>
</svg>"""
    return svg


def svg_to_data_url(svg_str):
    encoded = base64.b64encode(svg_str.encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def render_ytd_html(ytd_df):
    rows = []
    for _, row in ytd_df.iterrows():
        ytd_val = row["YTD Return (%)"]
        color   = "#1a9850" if ytd_val >= 0 else "#d73027"
        svg     = make_price_bar_svg(
            row["YTD Low"], row["Year Start"], row["Current"], row["YTD High"],
            inline=True,
        )
        is_benchmark = row["Ticker"] in ("SPY", "RSP")
        bg        = ""
        txt_style = "font-weight:bold;font-size:15px;" if is_benchmark else ""
        rows.append(f"""
        <tr style="border-bottom:1px solid rgba(128,128,128,0.2);{bg}">
          <td style="padding:0 12px;white-space:nowrap;{txt_style}">{row['Name']}</td>
          <td style="padding:0 12px;font-family:monospace;{txt_style}">{row['Ticker']}</td>
          <td style="padding:0 12px;text-align:right;color:{color};font-weight:bold;white-space:nowrap;{txt_style}">
            {ytd_val:+.2f}%</td>
          <td style="padding:0 12px">{svg}</td>
        </tr>""")

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:sans-serif">
      <thead>
        <tr style="border-bottom:2px solid rgba(128,128,128,0.4)">
          <th style="padding:8px 12px;text-align:left;white-space:nowrap">Name</th>
          <th style="padding:8px 12px;text-align:left">Ticker</th>
          <th style="padding:8px 12px;text-align:right;white-space:nowrap">YTD Return</th>
          <th style="padding:8px 12px;text-align:left">
            YTD Price Range &nbsp;&#8212;&nbsp; Low &nbsp;|&nbsp; &#9670; Year-Start &nbsp;|&nbsp; &#9679; Current &nbsp;|&nbsp; High
          </th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def fetch_all_data():
    year = date.today().year
    today = date.today()
    start = f"{year - 1}-12-28"

    symbols = list(TICKERS.keys())
    ohlcv   = yf.download(symbols, start=start, end=(today + timedelta(days=1)).isoformat(), auto_adjust=True, progress=False)
    raw     = ohlcv["Close"]
    raw_h   = ohlcv["High"]
    raw_l   = ohlcv["Low"]

    latest = raw.iloc[-1]
    as_of  = raw.index[-1].date()

    # ── YTD returns ───────────────────────────────────────────────────────────
    base = raw[raw.index.year == year - 1].iloc[-1]
    ytd  = ((latest - base) / base * 100).round(2)
    ytd_df = pd.DataFrame(ytd).reset_index()
    ytd_df.columns = ["Ticker", "YTD Return (%)"]
    ytd_df["Name"]       = ytd_df["Ticker"].map(TICKERS)
    ytd_df["As of Date"] = as_of

    # Merge in YTD price stats needed for the price bar
    ytd_raw    = raw[raw.index.year == year]
    price_stats = pd.DataFrame({
        "Ticker":     symbols,
        "Year Start": base[symbols].round(2).values,
        "YTD High":   ytd_raw[symbols].max().round(2).values,
        "YTD Low":    ytd_raw[symbols].min().round(2).values,
        "Current":    latest[symbols].round(2).values,
    })
    ytd_df = ytd_df.merge(price_stats, on="Ticker")
    ytd_df = ytd_df.sort_values("YTD Return (%)", ascending=False).reset_index(drop=True)

    # ── 52-week high / low ────────────────────────────────────────────────────
    window   = raw.tail(252)
    window_h = raw_h.tail(252)
    window_l = raw_l.tail(252)

    wk52_high = window[symbols].max()
    wk52_low  = window[symbols].min()

    # (52-wk high − 52-wk low) / current price
    hl_range_ratio = ((wk52_high - wk52_low) / latest[symbols]).round(4)

    # Parkinson volatility (annualised): σ = sqrt(1/(4·ln2·n) · Σ ln(H/L)²) · sqrt(252)
    log_hl   = np.log(window_h[symbols] / window_l[symbols])
    park_vol = (np.sqrt((log_hl ** 2).mean() / (4 * np.log(2)) * 252) * 100).round(2)

    price_df = pd.DataFrame({
        "Ticker":        symbols,
        "Current Price": latest[symbols].round(2).values,
        "52-Wk High":    wk52_high.round(2).values,
        "52-Wk Low":     wk52_low.round(2).values,
    })
    price_df["% from High"]      = ((price_df["Current Price"] - price_df["52-Wk High"]) / price_df["52-Wk High"] * 100).round(2)
    price_df["% from Low"]       = ((price_df["Current Price"] - price_df["52-Wk Low"])  / price_df["52-Wk Low"]  * 100).round(2)
    price_df["HL / Price"]       = hl_range_ratio.values
    price_df["Parkinson Vol (%)"]= park_vol.values
    price_df["Name"] = price_df["Ticker"].map(TICKERS)
    price_df = price_df.merge(ytd_df[["Ticker", "YTD Return (%)"]], on="Ticker")
    price_df = price_df.sort_values("YTD Return (%)", ascending=False).reset_index(drop=True)
    price_df = price_df[["Name", "Ticker", "Current Price", "52-Wk High", "52-Wk Low",
                          "% from High", "% from Low", "HL / Price", "Parkinson Vol (%)"]]

    # ── Daily YTD return series ───────────────────────────────────────────────
    ytd_series = raw[raw.index.year == year].copy()
    ytd_series = ((ytd_series - base) / base * 100).round(4)
    ytd_series = ytd_series.rename(columns=TICKERS)
    ytd_series.index = ytd_series.index.date

    return ytd_df, price_df, ytd_series


# ── Sector deep-dive helpers ──────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_constituents():
    gics_to_etf = {v: k for k, v in ETF_TO_GICS.items()}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
def get_sector_stock_data(tickers_tuple, year):
    tickers = list(tickers_tuple)
    dl      = tickers + ([] if "SPY" in tickers else ["SPY"])
    start   = f"{year - 1}-12-28"
    end     = (date.today() + timedelta(days=1)).isoformat()

    ohlcv = yf.download(dl, start=start, end=end, auto_adjust=True, progress=False)
    close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

    # Ensure DataFrame even for single ticker
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        high  = high.to_frame(tickers[0])
        low   = low.to_frame(tickers[0])

    latest    = close.iloc[-1]
    base      = close[close.index.year == year - 1].iloc[-1]
    ytd       = ((latest - base) / base * 100).round(2)
    wc        = close[tickers].tail(252)
    wh        = high[tickers].tail(252)
    wl        = low[tickers].tail(252)
    hi52      = wc.max()
    lo52      = wc.min()

    # Beta vs SPY from 252-day price returns
    rets    = close.pct_change().dropna()
    spy_var = rets["SPY"].var() if "SPY" in rets.columns else None
    betas   = {}
    if spy_var and spy_var > 0:
        for t in tickers:
            if t in rets.columns:
                betas[t] = round(rets[t].cov(rets["SPY"]) / spy_var, 2)

    # Parkinson volatility (annualised %)
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
            "Ticker":            t,
            "Current Price":     round(cur, 2),
            "YTD Return (%)":    round(float(ytd.get(t, np.nan)), 2),
            "52-Wk High":        round(h52, 2),
            "52-Wk Low":         round(l52, 2),
            "% from High":       round((cur - h52) / h52 * 100, 2) if h52 else None,
            "% from Low":        round((cur - l52) / l52 * 100, 2) if l52 else None,
            "Beta (1Y)":         betas.get(t),
            "Parkinson Vol (%)": park.get(t),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def get_sector_fundamentals(tickers_tuple):
    def fetch_one(ticker):
        try:
            info = yf.Ticker(ticker).info
            mc   = info.get("marketCap")
            pe   = info.get("trailingPE")
            fpe  = info.get("forwardPE")
            dy   = info.get("dividendYield")
            eg   = info.get("earningsGrowth")
            return {
                "Ticker":          ticker,
                "Mkt Cap ($B)":    round(mc / 1e9, 2)   if mc  else None,
                "P/E":             round(pe, 1)          if pe  else None,
                "Fwd P/E":         round(fpe, 1)         if fpe else None,
                "Div Yield (%)":   round(dy, 2)          if dy  else None,
                "EPS Growth (%)":  round(eg * 100, 1)    if eg  else None,
            }
        except Exception:
            return {"Ticker": ticker}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_one, list(tickers_tuple)))
    return pd.DataFrame(results)


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("S&P 500 Sector ETF — YTD Returns")
st.caption(f"Data as of {date.today().strftime('%B %d, %Y')}")

col_btn, _ = st.columns([1, 9])
with col_btn:
    if st.button("Refresh Data", type="primary"):
        st.cache_data.clear()

try:
    df, price_df, ytd_series = fetch_all_data()

    # ── YTD bar chart ─────────────────────────────────────────────────────────
    benchmarks = {TICKERS["SPY"], TICKERS["RSP"]}
    bar_texts = [
        f"<b>{v:+.2f}%</b>" if name in benchmarks else f"{v:+.2f}%"
        for name, v in zip(df["Name"], df["YTD Return (%)"])
    ]
    bar_text_sizes = [15 if name in benchmarks else 11 for name in df["Name"]]

    fig = px.bar(
        df,
        x="Name",
        y="YTD Return (%)",
        color="YTD Return (%)",
        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
        color_continuous_midpoint=0,
        text=bar_texts,
        title="YTD Return by Sector ETF",
        height=500,
    )
    fig.update_traces(textposition="outside", textfont=dict(size=bar_text_sizes))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="YTD Return (%)",
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── YTD line chart ────────────────────────────────────────────────────────
    st.subheader("YTD Performance Over Time")
    all_names = list(ytd_series.columns)

    if "line_chart_selection" not in st.session_state:
        st.session_state.line_chart_selection = {name: True for name in all_names}

    btn_col1, btn_col2, _ = st.columns([1, 1, 8])
    if btn_col1.button("Select All"):
        for name in all_names:
            st.session_state.line_chart_selection[name] = True
    if btn_col2.button("Clear All"):
        for name in all_names:
            st.session_state.line_chart_selection[name] = False

    chk_cols = st.columns(4)
    for i, name in enumerate(all_names):
        st.session_state.line_chart_selection[name] = chk_cols[i % 4].checkbox(
            name, value=st.session_state.line_chart_selection[name], key=f"chk_{name}"
        )

    selected = [n for n, v in st.session_state.line_chart_selection.items() if v]
    if selected:
        long_df = (
            ytd_series[selected]
            .reset_index()
            .rename(columns={"index": "Date"})
            .melt(id_vars="Date", var_name="ETF", value_name="YTD Return (%)")
        )
        spy_name = TICKERS["SPY"]
        line_fig = px.line(long_df, x="Date", y="YTD Return (%)", color="ETF",
                           title="YTD Return (%) — Daily", height=500)
        line_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        line_fig.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
        if spy_name in selected:
            line_fig.update_traces(
                selector={"name": spy_name},
                line=dict(color="black", width=2.5),
            )
        st.plotly_chart(line_fig, use_container_width=True)
    else:
        st.info("Select at least one ETF above to display the chart.")

    # ── Group Sector Performance ───────────────────────────────────────────────
    st.subheader("Group Sector Performance")

    group_df = pd.DataFrame({
        grp: ytd_series[[TICKERS[t] for t in tks]].mean(axis=1)
        for grp, tks in GROUPS.items()
    })
    group_df["SPY"] = ytd_series[TICKERS["SPY"]]
    group_df.index  = pd.to_datetime(group_df.index)

    smooth = st.slider("Smoothing (rolling days)", 1, 20, 1, key="grp_smooth")
    plot_gdf = group_df.rolling(smooth).mean() if smooth > 1 else group_df.copy()

    tab_ret, tab_spread = st.tabs(["Group Returns", "Rotation Signals"])

    with tab_ret:
        # Current-value metrics row
        latest_g = group_df.iloc[-1]
        prev5_g  = group_df.iloc[-6] if len(group_df) > 5 else group_df.iloc[0]
        mc = st.columns(4)
        for i, grp in enumerate(GROUP_COLORS):
            mc[i].metric(grp, f"{latest_g[grp]:+.2f}%",
                         delta=f"{latest_g[grp] - prev5_g[grp]:+.2f}% (5d)")

        long_gdf = (
            plot_gdf.reset_index()
            .rename(columns={"index": "Date"})
            .melt(id_vars="Date", var_name="Group", value_name="YTD Return (%)")
        )
        fig_g = px.line(long_gdf, x="Date", y="YTD Return (%)", color="Group",
                        color_discrete_map=GROUP_COLORS, height=460)
        fig_g.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_g.update_layout(legend_title="", yaxis_ticksuffix="%", xaxis_title="")
        st.plotly_chart(fig_g, use_container_width=True)

        # Per-group sector breakdown
        for grp, tickers in GROUPS.items():
            names = [TICKERS[t] for t in tickers]
            vals  = ytd_series[names].iloc[-1].round(2)
            parts = " · ".join(f"{t}: {v:+.2f}%" for t, v in zip(tickers, vals.values))
            st.caption(f"**{grp}:** {parts}")

    with tab_spread:
        cyc_def_now = group_df["Cyclical"].iloc[-1] - group_df["Defensive"].iloc[-1]
        sen_def_now = group_df["Sensitive"].iloc[-1] - group_df["Defensive"].iloc[-1]
        signal_lbl  = "Risk-On 🟢" if cyc_def_now > 0 else "Risk-Off 🔴"

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Market Signal (Cyc vs Def)", signal_lbl)
        sc2.metric("Cyclical − Defensive",  f"{cyc_def_now:+.2f}%")
        sc3.metric("Sensitive − Defensive", f"{sen_def_now:+.2f}%")

        spreads = {
            "Cyclical − Defensive":  (plot_gdf["Cyclical"]  - plot_gdf["Defensive"], "#3b82f6"),
            "Sensitive − Defensive": (plot_gdf["Sensitive"] - plot_gdf["Defensive"], "#f59e0b"),
        }
        fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=list(spreads.keys()),
                              vertical_spacing=0.10)
        for row, (label, (series, line_color)) in enumerate(spreads.items(), start=1):
            dates = series.index
            y     = series.values
            # Green fill above zero, red fill below zero
            fig_s.add_trace(go.Scatter(x=dates, y=np.where(y > 0, y, 0),
                                       fill="tozeroy", fillcolor="rgba(34,197,94,0.20)",
                                       line=dict(width=0), showlegend=False, hoverinfo="skip"),
                            row=row, col=1)
            fig_s.add_trace(go.Scatter(x=dates, y=np.where(y < 0, y, 0),
                                       fill="tozeroy", fillcolor="rgba(239,68,68,0.20)",
                                       line=dict(width=0), showlegend=False, hoverinfo="skip"),
                            row=row, col=1)
            fig_s.add_trace(go.Scatter(x=dates, y=y, name=label,
                                       line=dict(color=line_color, width=1.8)),
                            row=row, col=1)
            fig_s.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6,
                            row=row, col=1)

        fig_s.update_yaxes(ticksuffix="%")
        fig_s.update_layout(height=500, showlegend=False, xaxis_title="")
        st.plotly_chart(fig_s, use_container_width=True)

        st.caption(
            "**Reading the chart:** Above zero = cyclical/sensitive sectors outperform defensives (risk-on). "
            "Below zero = defensive rotation (risk-off). "
            "Green shading marks risk-on periods, red marks risk-off."
        )

    def color_return(val):
        color = "#1a9850" if val > 0 else "#d73027" if val < 0 else "gray"
        return f"color: {color}; font-weight: bold"

    # ── YTD Returns table (with price bar) ───────────────────────────────────
    st.subheader("YTD Returns")

    st.markdown(render_ytd_html(df), unsafe_allow_html=True)

    # ── Price & 52-week range table ───────────────────────────────────────────
    st.subheader("Price & 52-Week Range")

    def highlight_benchmark(row):
        bg = "background-color: rgba(99,102,241,0.15)" if row["Ticker"] in ("SPY", "RSP") else ""
        return [bg] * len(row)

    styled_price = (
        price_df.style
        .apply(highlight_benchmark, axis=1)
        .map(color_return, subset=["% from High", "% from Low"])
        .format({
            "Current Price":      "${:.2f}",
            "52-Wk High":         "${:.2f}",
            "52-Wk Low":          "${:.2f}",
            "% from High":        "{:+.2f}%",
            "% from Low":         "{:+.2f}%",
            "HL / Price":         "{:.2%}",
            "Parkinson Vol (%)":  "{:.2f}%",
        })
        .hide(axis="index")
    )
    st.dataframe(styled_price, use_container_width=True, height=490)

    st.caption(
        f"Prices as of {df['As of Date'].iloc[0]}. "
        f"YTD return calculated from the last trading day of {df['As of Date'].iloc[0].year - 1}. "
        f"52-week range based on last 252 trading days. "
        f"Price bar: gray=YTD range, colored fill=year-start→current, white tick=year-start, dot=current price."
    )

    # ── Sector Deep Dive ──────────────────────────────────────────────────────
    st.subheader("Sector Deep Dive — Stock Rankings")

    try:
        sp500 = get_sp500_constituents()
    except Exception as e:
        st.error(f"Could not load S&P 500 constituents: {e}")
        sp500 = None

    if sp500 is not None:
        sector_etfs   = [e for e in TICKERS if e not in ("SPY", "RSP")]
        sector_labels = [TICKERS[e] for e in sector_etfs]

        st.caption("Select a sector:")
        chosen_label = st.radio("", sector_labels,
                                horizontal=True, key="sector_radio",
                                label_visibility="collapsed")
        chosen_etf   = sector_etfs[sector_labels.index(chosen_label)]
        gics_name    = ETF_TO_GICS[chosen_etf]

        sector_stocks = sp500[sp500["ETF"] == chosen_etf]["Ticker"].tolist()
        st.caption(f"**{gics_name}** · {chosen_etf} · {len(sector_stocks)} constituents")

        if sector_stocks:
            tickers_t = tuple(sorted(sector_stocks))
            name_map  = sp500.set_index("Ticker")[["Name", "Sub-Industry"]].to_dict("index")

            with st.spinner("Loading price data…"):
                price_data = get_sector_stock_data(tickers_t, date.today().year)

            data = price_data.copy()
            data["Name"]         = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Name", t))
            data["Sub-Industry"] = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Sub-Industry", ""))
            data = data.sort_values("YTD Return (%)", ascending=False).reset_index(drop=True)
            data.insert(0, "Rank", range(1, len(data) + 1))

            tab_ind, tab_stocks = st.tabs(["Sub-Industry Summary", "Stock Rankings"])

            # ── Sub-Industry Summary ──────────────────────────────────────────
            with tab_ind:
                grp = (
                    data.groupby("Sub-Industry")
                    .agg(
                        Stocks        = ("Ticker",        "count"),
                        Avg_YTD       = ("YTD Return (%)", "mean"),
                        Median_YTD    = ("YTD Return (%)", "median"),
                        Pct_Positive  = ("YTD Return (%)", lambda x: (x > 0).mean() * 100),
                        Avg_Beta      = ("Beta (1Y)",      "mean"),
                        Avg_ParkVol   = ("Parkinson Vol (%)","mean"),
                    )
                    .reset_index()
                )
                # Best / Worst stock per sub-industry
                best = (data.loc[data.groupby("Sub-Industry")["YTD Return (%)"].idxmax(),
                                 ["Sub-Industry", "Ticker", "YTD Return (%)"]]
                           .rename(columns={"Ticker": "Best", "YTD Return (%)": "Best Ret (%)"}))
                worst = (data.loc[data.groupby("Sub-Industry")["YTD Return (%)"].idxmin(),
                                  ["Sub-Industry", "Ticker", "YTD Return (%)"]]
                            .rename(columns={"Ticker": "Worst", "YTD Return (%)": "Worst Ret (%)"}))
                grp = grp.merge(best, on="Sub-Industry").merge(worst, on="Sub-Industry")
                grp = grp.sort_values("Avg_YTD", ascending=False).reset_index(drop=True)
                grp.insert(0, "Rank", range(1, len(grp) + 1))
                grp.columns = ["Rank", "Sub-Industry", "# Stocks",
                                "Avg YTD (%)", "Median YTD (%)", "% Positive",
                                "Avg Beta", "Avg Vol (%)",
                                "Best", "Best Ret (%)", "Worst", "Worst Ret (%)"]

                # Bar chart — sub-industries by avg YTD
                si_chart = grp.sort_values("Avg YTD (%)")
                fig_si = px.bar(
                    si_chart, x="Avg YTD (%)", y="Sub-Industry", orientation="h",
                    color="Avg YTD (%)",
                    color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                    color_continuous_midpoint=0,
                    text=si_chart["Avg YTD (%)"].apply(lambda v: f"{v:+.2f}%"),
                    title=f"{chosen_label} — Sub-Industry Avg YTD Return",
                    height=max(380, len(grp) * 32),
                )
                fig_si.update_traces(textposition="outside")
                fig_si.update_layout(coloraxis_showscale=False,
                                     xaxis_ticksuffix="%", yaxis_title="",
                                     xaxis_title="Avg YTD Return (%)")
                st.plotly_chart(fig_si, use_container_width=True)

                # Summary table
                fmt_si = {
                    "Avg YTD (%)":    "{:+.2f}%",
                    "Median YTD (%)": "{:+.2f}%",
                    "% Positive":     "{:.0f}%",
                    "Avg Beta":       "{:.2f}",
                    "Avg Vol (%)":    "{:.2f}%",
                    "Best Ret (%)":   "{:+.2f}%",
                    "Worst Ret (%)":  "{:+.2f}%",
                }
                styled_si = (
                    grp.style
                    .map(color_return, subset=["Avg YTD (%)", "Median YTD (%)",
                                               "Best Ret (%)", "Worst Ret (%)"])
                    .format(fmt_si, na_rep="—")
                    .hide(axis="index")
                )
                st.dataframe(styled_si, use_container_width=True,
                             height=min(35 * len(grp) + 40, 600))

            # ── Stock Rankings ────────────────────────────────────────────────
            with tab_stocks:
                chart_orient = st.radio("Chart orientation:", ["Horizontal", "Vertical"],
                                        horizontal=True, key="chart_orient",
                                        label_visibility="visible")
                if chart_orient == "Horizontal":
                    ranked = data.sort_values("YTD Return (%)")
                    fig_r = px.bar(
                        ranked, x="YTD Return (%)", y="Ticker", orientation="h",
                        color="YTD Return (%)",
                        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                        color_continuous_midpoint=0,
                        text=ranked["YTD Return (%)"].apply(lambda v: f"{v:+.2f}%"),
                        hover_data={"Name": True, "YTD Return (%)": False},
                        title=f"{chosen_label} — YTD Return Ranking",
                        height=max(420, len(data) * 22),
                    )
                    fig_r.update_traces(textposition="outside")
                    fig_r.update_layout(coloraxis_showscale=False,
                                        xaxis_ticksuffix="%", yaxis_title="",
                                        xaxis_title="YTD Return (%)")
                else:
                    ranked = data.sort_values("YTD Return (%)", ascending=False)
                    fig_r = px.bar(
                        ranked, x="Ticker", y="YTD Return (%)", orientation="v",
                        color="YTD Return (%)",
                        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                        color_continuous_midpoint=0,
                        text=ranked["YTD Return (%)"].apply(lambda v: f"{v:+.2f}%"),
                        hover_data={"Name": True, "YTD Return (%)": False},
                        title=f"{chosen_label} — YTD Return Ranking",
                        height=500,
                    )
                    fig_r.update_traces(textposition="outside")
                    fig_r.update_layout(coloraxis_showscale=False,
                                        yaxis_ticksuffix="%", xaxis_title="",
                                        yaxis_title="YTD Return (%)")
                st.plotly_chart(fig_r, use_container_width=True)

                # Fundamental factors toggle (after chart)
                show_fund = st.toggle("Show fundamental factors", value=True, key="show_fund")

                if show_fund:
                    with st.spinner("Loading fundamentals (cached daily)…"):
                        fund_data = get_sector_fundamentals(tickers_t)
                    data = data.merge(fund_data, on="Ticker", how="left")

                base_cols = ["Rank", "Ticker", "Name", "Sub-Industry",
                             "YTD Return (%)", "Current Price",
                             "52-Wk High", "52-Wk Low", "% from High", "% from Low",
                             "Beta (1Y)", "Parkinson Vol (%)"]
                fund_cols = ["Mkt Cap ($B)", "P/E", "Fwd P/E", "Div Yield (%)", "EPS Growth (%)"]
                cols = base_cols + (fund_cols if show_fund else [])
                cols = [c for c in cols if c in data.columns]
                data = data[cols]

                fmt = {
                    "YTD Return (%)":    "{:+.2f}%",
                    "Current Price":     "${:.2f}",
                    "52-Wk High":        "${:.2f}",
                    "52-Wk Low":         "${:.2f}",
                    "% from High":       "{:+.2f}%",
                    "% from Low":        "{:+.2f}%",
                    "Beta (1Y)":         "{:.2f}",
                    "Parkinson Vol (%)": "{:.2f}%",
                    "Mkt Cap ($B)":      "${:.1f}B",
                    "P/E":               "{:.1f}x",
                    "Fwd P/E":           "{:.1f}x",
                    "Div Yield (%)":     "{:.2f}%",
                    "EPS Growth (%)":    "{:+.1f}%",
                }
                fmt = {k: v for k, v in fmt.items() if k in data.columns}

                styled_stocks = (
                    data.style
                    .map(color_return, subset=["YTD Return (%)", "% from High", "% from Low"])
                    .format(fmt, na_rep="—")
                    .hide(axis="index")
                )
                st.dataframe(styled_stocks, use_container_width=True,
                             height=min(35 * len(data) + 40, 800))

except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("Please check your internet connection and try refreshing.")
