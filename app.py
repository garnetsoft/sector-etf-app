import io

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

from utils import ETF_TO_GICS, make_price_bar_svg, svg_to_data_url, get_sp500_constituents

st.set_page_config(page_title="S&P 500 Sector ETF Returns", layout="wide")

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


def render_ytd_html(ytd_df):
    has_extra = "52-Wk High" in ytd_df.columns

    def _fmt_price(v):
        try:
            return f"${float(v):.2f}" if v == v else "—"
        except Exception:
            return "—"

    def _fmt_pct(v, plus=False):
        try:
            f = float(v)
            return ("—" if f != f else (f"{f:+.2f}%" if plus else f"{f:.2f}%"))
        except Exception:
            return "—"

    def _fmt_ratio(v):
        try:
            f = float(v)
            return "—" if f != f else f"{f:.2%}"
        except Exception:
            return "—"

    def _pct_color(v):
        try:
            f = float(v)
            return "#1a9850" if f >= 0 else "#d73027"
        except Exception:
            return "#888"

    rows = []
    for _, row in ytd_df.iterrows():
        ytd_val = row["Return (%)"]
        color   = "#1a9850" if ytd_val >= 0 else "#d73027"
        svg     = make_price_bar_svg(
            row["Period Low"], row["Year Start"], row["Current"], row["Period High"],
            inline=True,
        )
        is_benchmark = row["Ticker"] in ("SPY", "RSP")
        txt_style = "font-weight:bold;font-size:15px;" if is_benchmark else ""

        extra_tds = ""
        if has_extra:
            hlp = row.get("Period HL/Price")
            pv  = row.get("Parkinson Vol (%)")
            extra_tds = (
                f'<td style="padding:6px 12px;text-align:right;width:90px;font-family:monospace;white-space:nowrap">{_fmt_ratio(hlp)}</td>'
                f'<td style="padding:6px 12px;text-align:right;width:90px;font-family:monospace;white-space:nowrap">{_fmt_pct(pv)}</td>'
                f'<td style="padding:6px 12px;text-align:right;width:90px;font-family:monospace;white-space:nowrap">{_fmt_price(row.get("52-Wk Low"))}</td>'
                f'<td style="padding:6px 12px;text-align:right;width:90px;font-family:monospace;white-space:nowrap">{_fmt_price(row.get("52-Wk High"))}</td>'
            )

        rows.append(f"""
        <tr style="border-bottom:1px solid rgba(128,128,128,0.2);">
          <td style="padding:6px 12px;width:250px;max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;{txt_style}">{row['Name']}</td>
          <td style="padding:6px 12px;width:90px;font-family:monospace;{txt_style}">{row['Ticker']}</td>
          <td style="padding:6px 12px;text-align:right;color:{color};font-weight:bold;white-space:nowrap;{txt_style}">{ytd_val:+.2f}%</td>
          <td style="padding:6px 12px;width:680px;min-width:680px">{svg}</td>{extra_tds}
        </tr>""")

    extra_headers = ""
    if has_extra:
        period = ytd_df.attrs.get("period_label", "Period")
        extra_headers = (
            f'<th style="padding:8px 12px;text-align:right;width:90px;white-space:nowrap">HL/Price ({period})</th>'
            f'<th style="padding:8px 12px;text-align:right;width:90px;white-space:nowrap">Parkinson Vol ({period})</th>'
            '<th style="padding:8px 12px;text-align:right;width:90px;white-space:nowrap">52-Wk Low</th>'
            '<th style="padding:8px 12px;text-align:right;width:90px;white-space:nowrap">52-Wk High</th>'
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:sans-serif">
      <thead>
        <tr style="border-bottom:2px solid rgba(128,128,128,0.4)">
          <th style="padding:8px 12px;text-align:left;width:250px;max-width:250px">Name</th>
          <th style="padding:8px 12px;text-align:left;width:90px">Ticker</th>
          <th style="padding:8px 12px;text-align:right;width:90px;white-space:nowrap">Return</th>
          <th style="padding:8px 12px;text-align:left">
            Price Range &nbsp;&#8212;&nbsp; Low &nbsp;|&nbsp; &#9670; Period-Start &nbsp;|&nbsp; &#9679; Current &nbsp;|&nbsp; High
          </th>{extra_headers}
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def fetch_all_data(start_date: date, end_date: date):
    symbols     = list(TICKERS.keys())
    fetch_start = (start_date - timedelta(days=10)).isoformat()
    fetch_end   = (end_date   + timedelta(days=1)).isoformat()

    ohlcv = yf.download(symbols, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    raw   = ohlcv["Close"]
    raw_h = ohlcv["High"]
    raw_l = ohlcv["Low"]

    # Base price = last close on or before start_date
    base_raw = raw[raw.index.date <= start_date]
    base     = base_raw.iloc[-1]

    # Period = start_date through end_date
    period_raw = raw[(raw.index.date >= start_date) & (raw.index.date <= end_date)]
    latest     = period_raw.iloc[-1]
    as_of      = period_raw.index[-1].date()

    # ── Period returns ────────────────────────────────────────────────────────
    ret    = ((latest - base) / base * 100).round(2)
    ytd_df = pd.DataFrame(ret).reset_index()
    ytd_df.columns = ["Ticker", "Return (%)"]
    ytd_df["Name"]       = ytd_df["Ticker"].map(TICKERS)
    ytd_df["As of Date"] = as_of

    price_stats = pd.DataFrame({
        "Ticker":     symbols,
        "Year Start": base[symbols].round(2).values,
        "Period High":   period_raw[symbols].max().round(2).values,
        "Period Low":    period_raw[symbols].min().round(2).values,
        "Current":    latest[symbols].round(2).values,
    })
    ytd_df = ytd_df.merge(price_stats, on="Ticker")
    ytd_df = ytd_df.sort_values("Return (%)", ascending=False).reset_index(drop=True)

    # ── 52-week high / low (always trailing 252 days from end_date) ───────────
    window   = raw[raw.index.date <= end_date].tail(252)
    window_h = raw_h[raw_h.index.date <= end_date].tail(252)
    window_l = raw_l[raw_l.index.date <= end_date].tail(252)

    wk52_high      = window[symbols].max()
    wk52_low       = window[symbols].min()
    hl_range_ratio = ((wk52_high - wk52_low) / latest[symbols]).round(4)

    log_hl   = np.log(window_h[symbols] / window_l[symbols])
    park_vol = (np.sqrt((log_hl ** 2).mean() / (4 * np.log(2)) * 252) * 100).round(2)

    price_df = pd.DataFrame({
        "Ticker":        symbols,
        "Current Price": latest[symbols].round(2).values,
        "52-Wk High":    wk52_high.round(2).values,
        "52-Wk Low":     wk52_low.round(2).values,
    })
    price_df["% from High"]       = ((price_df["Current Price"] - price_df["52-Wk High"]) / price_df["52-Wk High"] * 100).round(2)
    price_df["% from Low"]        = ((price_df["Current Price"] - price_df["52-Wk Low"])  / price_df["52-Wk Low"]  * 100).round(2)
    price_df["HL / Price"]        = hl_range_ratio.values
    price_df["Parkinson Vol (%)"] = park_vol.values
    price_df["Name"]              = price_df["Ticker"].map(TICKERS)
    price_df = price_df.merge(ytd_df[["Ticker", "Return (%)"]], on="Ticker")
    price_df = price_df.sort_values("Return (%)", ascending=False).reset_index(drop=True)
    price_df = price_df[["Name", "Ticker", "Return (%)", "Current Price", "52-Wk High", "52-Wk Low",
                          "% from High", "% from Low", "HL / Price", "Parkinson Vol (%)"]]

    # ── Daily return series ───────────────────────────────────────────────────
    ytd_series = period_raw.copy()
    ytd_series = ((ytd_series - base) / base * 100).round(4)
    ytd_series = ytd_series.rename(columns=TICKERS)
    ytd_series.index = ytd_series.index.date

    return ytd_df, price_df, ytd_series


# ── Sector deep-dive helpers ──────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_stock_data(tickers_tuple, start_date: date, end_date: date):
    tickers     = list(tickers_tuple)
    dl          = tickers + ([] if "SPY" in tickers else ["SPY"])
    fetch_start = (start_date - timedelta(days=10)).isoformat()
    fetch_end   = (end_date   + timedelta(days=1)).isoformat()

    ohlcv = yf.download(dl, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    close, high, low = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]

    # Ensure DataFrame even for single ticker
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        high  = high.to_frame(tickers[0])
        low   = low.to_frame(tickers[0])

    base      = close[close.index.date <= start_date].iloc[-1]
    period_c  = close[(close.index.date >= start_date) & (close.index.date <= end_date)]
    latest    = period_c.iloc[-1]
    ytd       = ((latest - base) / base * 100).round(2)
    wc        = close[tickers][close.index.date <= end_date].tail(252)
    wh        = high[tickers][high.index.date   <= end_date].tail(252)
    wl        = low[tickers][low.index.date     <= end_date].tail(252)
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
            "Return (%)":    round(float(ytd.get(t, np.nan)), 2),
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
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def fetch_one(ticker):
        try:
            info = yf.Ticker(ticker, session=session).info
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

# Sidebar date range
today = date.today()

PRESETS = {
    "YTD":    date(today.year, 1, 1),
    "1Y":     today - timedelta(days=365),
    "2Y":     today - timedelta(days=365*2),
    "3Y":     today - timedelta(days=365*3),
    "5Y":     today - timedelta(days=365*5),
    "10Y":    today - timedelta(days=365*10),
    "Custom": None,
}

if "start_date_input" not in st.session_state:
    st.session_state.start_date_input = PRESETS["YTD"]

def on_preset_change():
    preset_val = PRESETS[st.session_state.preset_radio]
    if preset_val is not None:
        st.session_state.start_date_input = preset_val

def on_date_change():
    st.session_state.preset_radio = "Custom"

with st.sidebar:
    st.header("Date Range")
    start_date = st.date_input("Start Date", max_value=today, key="start_date_input", on_change=on_date_change)
    end_date   = st.date_input("End Date", value=today, min_value=start_date, key="end_date_input", on_change=on_date_change)
    st.radio("Quick Select", list(PRESETS.keys()), horizontal=True, index=0,
             key="preset_radio", on_change=on_preset_change)
    if st.session_state.get("preset_radio") == "Custom":
        st.caption("Set Start and End Date manually above.")
is_ytd = (start_date == date(today.year, 1, 1) and end_date == today)
period_label = f"{start_date.strftime('%m-%d-%Y')} --> {end_date.strftime('%m-%d-%Y')}"

st.title("S&P 500 Sector ETF — Returns")
st.caption(f"Period: **{start_date.strftime('%m-%d-%Y')}** --> **{end_date.strftime('%m-%d-%Y')}**")

col_btn, _ = st.columns([1, 9])
with col_btn:
    if st.button("Refresh Data", type="primary"):
        st.cache_data.clear()

try:
    df, price_df, ytd_series = fetch_all_data(start_date, end_date)

    # ── YTD bar chart ─────────────────────────────────────────────────────────
    benchmarks = {TICKERS["SPY"], TICKERS["RSP"]}
    bar_texts = [
        f"<b>{v:+.2f}%</b>" if name in benchmarks else f"{v:+.2f}%"
        for name, v in zip(df["Name"], df["Return (%)"])
    ]
    bar_text_sizes = [15 if name in benchmarks else 11 for name in df["Name"]]

    fig = px.bar(
        df,
        x="Name",
        y="Return (%)",
        color="Return (%)",
        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
        color_continuous_midpoint=0,
        text=bar_texts,
        title=f"{period_label} Return by Sector ETF",
        height=500,
    )
    fig.update_traces(textposition="outside", textfont=dict(size=bar_text_sizes))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Return (%)",
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, width='stretch')

    # ── YTD line chart ────────────────────────────────────────────────────────
    st.subheader(f"Performance Over Time — {period_label}")
    all_names = list(ytd_series.columns)

    if "line_chart_selection" not in st.session_state:
        st.session_state.line_chart_selection = {name: True for name in all_names}

    btn_col1, btn_col2, _ = st.columns([1, 1, 8])
    if btn_col1.button("Select All"):
        for name in all_names:
            st.session_state.line_chart_selection[name] = True
            st.session_state[f"chk_{name}"] = True
    if btn_col2.button("Clear All"):
        keep = {TICKERS["SPY"], TICKERS["RSP"]}
        for name in all_names:
            val = name in keep
            st.session_state.line_chart_selection[name] = val
            st.session_state[f"chk_{name}"] = val

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
            .melt(id_vars="Date", var_name="ETF", value_name="Return (%)")
        )
        spy_name = TICKERS["SPY"]
        line_fig = px.line(long_df, x="Date", y="Return (%)", color="ETF",
                           title=f"Return (%) — Daily ({period_label})", height=500)
        line_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        line_fig.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
        if spy_name in selected:
            line_fig.update_traces(
                selector={"name": spy_name},
                line=dict(color="black", width=2.5),
            )
        st.plotly_chart(line_fig, width='stretch')
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
            .melt(id_vars="Date", var_name="Group", value_name="Return (%)")
        )
        fig_g = px.line(long_gdf, x="Date", y="Return (%)", color="Group",
                        color_discrete_map=GROUP_COLORS, height=460)
        fig_g.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_g.update_layout(legend_title="", yaxis_ticksuffix="%", xaxis_title="")
        st.plotly_chart(fig_g, width='stretch')

        # Per-group sector breakdown
        max_sectors = max(len(t) for t in GROUPS.values())
        for grp, tickers in GROUPS.items():
            names = [TICKERS[t] for t in tickers]
            vals  = ytd_series[names].iloc[-1].round(2)
            label_col, *sector_cols = st.columns([1] + [1] * max_sectors)
            label_col.markdown(
                f"<div style='padding-top:6px;font-weight:bold;font-size:13px'>{grp}</div>",
                unsafe_allow_html=True,
            )
            for col, t, v in zip(sector_cols, tickers, vals.values):
                color = "#1a9850" if v > 0 else "#d73027"
                col.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:12px;color:gray'>{t}</div>"
                    f"<div style='font-size:15px;font-weight:bold;color:{color}'>{v:+.2f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

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
        st.plotly_chart(fig_s, width='stretch')

        st.caption(
            "**Reading the chart:** Above zero = cyclical/sensitive sectors outperform defensives (risk-on). "
            "Below zero = defensive rotation (risk-off). "
            "Green shading marks risk-on periods, red marks risk-off."
        )

    def color_return(val):
        color = "#1a9850" if val > 0 else "#d73027" if val < 0 else "gray"
        return f"color: {color}; font-weight: bold"

    # ── Combined Returns + Price table ───────────────────────────────────────
    st.subheader(f"Returns & Price — {period_label}")

    extra_cols = ["Ticker", "52-Wk High", "52-Wk Low", "Parkinson Vol (%)"]
    combined_df = df.merge(price_df[extra_cols], on="Ticker", how="left")
    combined_df["Period HL/Price"] = ((combined_df["Period High"] - combined_df["Period Low"]) / combined_df["Current"]).round(4)
    combined_df.attrs["period_label"] = st.session_state.get("preset_radio", "Period")
    st.markdown(render_ytd_html(combined_df), unsafe_allow_html=True)

    st.caption(
        f"Prices as of {df['As of Date'].iloc[0]}. "
        f"Return calculated from {start_date.strftime('%m-%d-%Y')} → {end_date.strftime('%m-%d-%Y')}. "
        f"52-week range based on last 252 trading days from end date. "
        f"Price bar: gray=period range, colored fill=period-start→current, ◆=period-start, ●=current price."
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
        chosen_label = st.radio("Select sector", sector_labels,
                                horizontal=True, key="sector_radio",
                                label_visibility="collapsed")
        chosen_etf   = sector_etfs[sector_labels.index(chosen_label)]
        gics_name    = ETF_TO_GICS[chosen_etf]

        sector_stocks = sp500[sp500["ETF"] == chosen_etf]["Ticker"].tolist()
        n_subindustries = sp500[sp500["ETF"] == chosen_etf]["Sub-Industry"].nunique()
        st.caption(f"**{gics_name}** · {chosen_etf} · {n_subindustries} sub-industries · {len(sector_stocks)} constituents")

        if sector_stocks:
            tickers_t = tuple(sorted(sector_stocks))
            name_map  = sp500.set_index("Ticker")[["Name", "Sub-Industry"]].to_dict("index")

            with st.spinner("Loading price data…"):
                price_data = get_sector_stock_data(tickers_t, start_date, end_date)

            data = price_data.copy()
            data["Name"]         = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Name", t))
            data["Sub-Industry"] = data["Ticker"].map(lambda t: name_map.get(t, {}).get("Sub-Industry", ""))
            data = data.sort_values("Return (%)", ascending=False).reset_index(drop=True)
            data.insert(0, "Rank", range(1, len(data) + 1))

            tab_ind, tab_stocks = st.tabs(["Sub-Industry Summary", "Stock Rankings"])

            # ── Sub-Industry Summary ──────────────────────────────────────────
            with tab_ind:
                grp = (
                    data.groupby("Sub-Industry")
                    .agg(
                        Stocks        = ("Ticker",        "count"),
                        Avg_Ret       = ("Return (%)", "mean"),
                        Median_Ret    = ("Return (%)", "median"),
                        Pct_Positive  = ("Return (%)", lambda x: (x > 0).mean() * 100),
                        Avg_Beta      = ("Beta (1Y)",      "mean"),
                        Avg_ParkVol   = ("Parkinson Vol (%)","mean"),
                    )
                    .reset_index()
                )
                # Best / Worst stock per sub-industry
                best = (data.loc[data.groupby("Sub-Industry")["Return (%)"].idxmax(),
                                 ["Sub-Industry", "Ticker", "Return (%)"]]
                           .rename(columns={"Ticker": "Best", "Return (%)": "Best Ret (%)"}))
                worst = (data.loc[data.groupby("Sub-Industry")["Return (%)"].idxmin(),
                                  ["Sub-Industry", "Ticker", "Return (%)"]]
                            .rename(columns={"Ticker": "Worst", "Return (%)": "Worst Ret (%)"}))
                grp = grp.merge(best, on="Sub-Industry").merge(worst, on="Sub-Industry")
                grp = grp.sort_values("Avg_Ret", ascending=False).reset_index(drop=True)
                grp.columns = ["Sub-Industry", "# Stocks",
                                "Avg Return (%)", "Median Return (%)", "% Positive",
                                "Avg Beta", "Avg Vol (%)",
                                "Best", "Best Ret (%)", "Worst", "Worst Ret (%)"]

                # Bar chart — sub-industries by avg YTD
                si_chart = grp.sort_values("Avg Return (%)")
                fig_si = px.bar(
                    si_chart, x="Avg Return (%)", y="Sub-Industry", orientation="h",
                    color="Avg Return (%)",
                    color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                    color_continuous_midpoint=0,
                    text=si_chart["Avg Return (%)"].apply(lambda v: f"{v:+.2f}%"),
                    title=f"{chosen_label} — Sub-Industry Avg Return ({period_label})",
                    height=max(380, len(grp) * 32),
                )
                fig_si.update_traces(textposition="outside")
                fig_si.update_layout(coloraxis_showscale=False,
                                     xaxis_ticksuffix="%", yaxis_title="",
                                     xaxis_title="Avg Return (%)")
                st.plotly_chart(fig_si, width='stretch')

                # Summary table
                fmt_si = {
                    "Avg Return (%)":    "{:+.2f}%",
                    "Median Return (%)": "{:+.2f}%",
                    "% Positive":     "{:.0f}%",
                    "Avg Beta":       "{:.2f}",
                    "Avg Vol (%)":    "{:.2f}%",
                    "Best Ret (%)":   "{:+.2f}%",
                    "Worst Ret (%)":  "{:+.2f}%",
                }
                styled_si = (
                    grp.style
                    .map(color_return, subset=["Avg Return (%)", "Median Return (%)",
                                               "Best Ret (%)", "Worst Ret (%)"])
                    .format(fmt_si, na_rep="—")
                    .hide(axis="index")
                )
                st.dataframe(styled_si, width='stretch',
                             height=38 * len(grp) + 50)

                # ── Commentary ───────────────────────────────────────────────
                n_si        = len(grp)
                n_positive  = (grp["Avg Return (%)"] > 0).sum()
                n_negative  = n_si - n_positive
                top         = grp.iloc[0]
                bottom      = grp.iloc[-1]
                spread      = top["Avg Return (%)"] - bottom["Avg Return (%)"]
                best_stock  = grp.loc[grp["Best Ret (%)"].idxmax()]
                worst_stock = grp.loc[grp["Worst Ret (%)"].idxmin()]
                high_bread  = grp[grp["% Positive"] >= 75]
                low_bread   = grp[grp["% Positive"] <= 25]

                breadth_line = (
                    f"{len(high_bread)} sub-industr{'y' if len(high_bread)==1 else 'ies'} "
                    f"with ≥75% of stocks positive"
                    if not high_bread.empty else
                    "no sub-industries with broad positive breadth (≥75%)"
                )
                low_bread_line = (
                    f"; {len(low_bread)} with ≤25% positive"
                    if not low_bread.empty else ""
                )

                commentary = f"""
**{gics_name} — Sub-Industry Performance Summary** &nbsp;·&nbsp; {period_label}

**{n_positive} of {n_si}** sub-industries positive &nbsp;|&nbsp; **{n_negative}** negative &nbsp;|&nbsp; spread: **{spread:+.2f}%**

- **Leader:** {top['Sub-Industry']} ({top['Avg Return (%)']:+.2f}% avg, {top['% Positive']:.0f}% of stocks positive) — best stock: **{top['Best']}** ({top['Best Ret (%)']:+.2f}%)
- **Laggard:** {bottom['Sub-Industry']} ({bottom['Avg Return (%)']:+.2f}% avg, {bottom['% Positive']:.0f}% of stocks positive) — worst stock: **{bottom['Worst']}** ({bottom['Worst Ret (%)']:+.2f}%)
- **Top individual performer:** {best_stock['Best']} ({best_stock['Best Ret (%)']:+.2f}%) in {best_stock['Sub-Industry']}
- **Worst individual performer:** {worst_stock['Worst']} ({worst_stock['Worst Ret (%)']:+.2f}%) in {worst_stock['Sub-Industry']}
- **Breadth:** {breadth_line}{low_bread_line}
"""
                st.markdown(commentary)

            # ── Stock Rankings ────────────────────────────────────────────────
            with tab_stocks:
                chart_orient = st.radio("Chart orientation:", ["Horizontal", "Vertical"],
                                        horizontal=True, key="chart_orient",
                                        label_visibility="visible")
                if chart_orient == "Horizontal":
                    ranked = data.sort_values("Return (%)")
                    fig_r = px.bar(
                        ranked, x="Return (%)", y="Ticker", orientation="h",
                        color="Return (%)",
                        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                        color_continuous_midpoint=0,
                        text=ranked["Return (%)"].apply(lambda v: f"{v:+.2f}%"),
                        hover_data={"Name": True, "Return (%)": False},
                        title=f"{chosen_label} — Return Ranking ({period_label})",
                        height=max(420, len(data) * 22),
                    )
                    fig_r.update_traces(textposition="outside")
                    fig_r.update_layout(coloraxis_showscale=False,
                                        xaxis_ticksuffix="%", yaxis_title="",
                                        xaxis_title="Return (%)")
                else:
                    ranked = data.sort_values("Return (%)", ascending=False)
                    fig_r = px.bar(
                        ranked, x="Ticker", y="Return (%)", orientation="v",
                        color="Return (%)",
                        color_continuous_scale=["#d73027", "#fee08b", "#1a9850"],
                        color_continuous_midpoint=0,
                        text=ranked["Return (%)"].apply(lambda v: f"{v:+.2f}%"),
                        hover_data={"Name": True, "Return (%)": False},
                        title=f"{chosen_label} — Return Ranking ({period_label})",
                        height=500,
                    )
                    fig_r.update_traces(textposition="outside")
                    fig_r.update_layout(coloraxis_showscale=False,
                                        yaxis_ticksuffix="%", xaxis_title="",
                                        yaxis_title="Return (%)")
                st.plotly_chart(fig_r, width='stretch')

                # Fundamental factors toggle (after chart)
                show_fund = st.toggle("Show fundamental factors", value=True, key="show_fund")

                if show_fund:
                    with st.spinner("Loading fundamentals (cached daily)…"):
                        fund_data = get_sector_fundamentals(tickers_t)
                    data = data.merge(fund_data, on="Ticker", how="left")

                base_cols = ["Rank", "Ticker", "Name", "Sub-Industry",
                             "Return (%)", "Current Price",
                             "52-Wk High", "52-Wk Low", "% from High", "% from Low",
                             "Beta (1Y)", "Parkinson Vol (%)"]
                fund_cols = ["Mkt Cap ($B)", "P/E", "Fwd P/E", "Div Yield (%)", "EPS Growth (%)"]
                cols = base_cols + (fund_cols if show_fund else [])
                cols = [c for c in cols if c in data.columns]
                data = data[cols]

                fmt = {
                    "Return (%)":    "{:+.2f}%",
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
                    .map(color_return, subset=["Return (%)", "% from High", "% from Low"])
                    .format(fmt, na_rep="—")
                    .hide(axis="index")
                )
                st.dataframe(styled_stocks, width='stretch',
                             height=35 * len(data) + 40, hide_index=True)

except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("Please check your internet connection and try refreshing.")
