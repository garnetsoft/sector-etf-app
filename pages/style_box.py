import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from utils import make_price_bar_svg

st.set_page_config(page_title="Equity Style Box", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

CAPS   = ["Large", "Mid", "Small"]
STYLES = ["Value", "Blend", "Growth"]

STYLE_COLORS = {"Value": "#3b82f6", "Blend": "#6b7280", "Growth": "#f59e0b"}
CAP_COLORS   = {"Large": "#1e3a5f", "Mid": "#2563eb", "Small": "#93c5fd"}

FUND_FAMILIES = {
    "Vanguard": {
        "style_box": {
            ("Large", "Value"):  "VTV",
            ("Large", "Blend"):  "VOO",
            ("Large", "Growth"): "VUG",
            ("Mid",   "Value"):  "VOE",
            ("Mid",   "Blend"):  "VO",
            ("Mid",   "Growth"): "VOT",
            ("Small", "Value"):  "VBR",
            ("Small", "Blend"):  "VB",
            ("Small", "Growth"): "VBK",
        },
        "etf_names": {
            "VTV": "Vanguard Value",
            "VOO": "Vanguard S&P 500",
            "VUG": "Vanguard Growth",
            "VOE": "Vanguard Mid Value",
            "VO":  "Vanguard Mid-Cap",
            "VOT": "Vanguard Mid Growth",
            "VBR": "Vanguard Small Value",
            "VB":  "Vanguard Small-Cap",
            "VBK": "Vanguard Small Growth",
        },
    },
    "SPDR": {
        "style_box": {
            ("Large", "Value"):  "SPYV",
            ("Large", "Blend"):  "SPY",
            ("Large", "Growth"): "SPYG",
            ("Mid",   "Value"):  "MDYV",
            ("Mid",   "Blend"):  "SPMD",
            ("Mid",   "Growth"): "MDYG",
            ("Small", "Value"):  "SLYV",
            ("Small", "Blend"):  "SPSM",
            ("Small", "Growth"): "SLYG",
        },
        "etf_names": {
            "SPYV": "SPDR Portfolio S&P 500 Value",
            "SPY":  "SPDR S&P 500",
            "SPYG": "SPDR Portfolio S&P 500 Growth",
            "MDYV": "SPDR S&P MidCap 400 Value",
            "SPMD": "SPDR Portfolio S&P 400 Mid Cap",
            "MDYG": "SPDR S&P MidCap 400 Growth",
            "SLYV": "SPDR S&P 600 Small Cap Value",
            "SPSM": "SPDR Portfolio S&P 600 Small Cap",
            "SLYG": "SPDR S&P 600 Small Cap Growth",
        },
    },
    "BlackRock": {
        "style_box": {
            ("Large", "Value"):  "IVE",
            ("Large", "Blend"):  "IVV",
            ("Large", "Growth"): "IVW",
            ("Mid",   "Value"):  "IJJ",
            ("Mid",   "Blend"):  "IJH",
            ("Mid",   "Growth"): "IJK",
            ("Small", "Value"):  "IJS",
            ("Small", "Blend"):  "IJR",
            ("Small", "Growth"): "IJT",
        },
        "etf_names": {
            "IVE": "iShares S&P 500 Value",
            "IVV": "iShares Core S&P 500",
            "IVW": "iShares S&P 500 Growth",
            "IJJ": "iShares S&P Mid-Cap 400 Value",
            "IJH": "iShares Core S&P Mid-Cap",
            "IJK": "iShares S&P Mid-Cap 400 Growth",
            "IJS": "iShares S&P Small-Cap 600 Value",
            "IJR": "iShares Core S&P Small-Cap",
            "IJT": "iShares S&P Small-Cap 600 Growth",
        },
    },
}


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def fetch_data(start_date: date, end_date: date, tickers_tuple: tuple, family: str):
    tickers   = list(tickers_tuple)
    style_box = FUND_FAMILIES[family]["style_box"]
    etf_names = FUND_FAMILIES[family]["etf_names"]

    fetch_start = (start_date - timedelta(days=10)).isoformat()
    fetch_end   = (end_date   + timedelta(days=1)).isoformat()

    ohlcv = yf.download(tickers, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    raw   = ohlcv["Close"]
    raw_h = ohlcv["High"]
    raw_l = ohlcv["Low"]

    base_raw   = raw[raw.index.date <= start_date]
    base       = base_raw.iloc[-1]
    period_raw = raw[(raw.index.date >= start_date) & (raw.index.date <= end_date)]
    latest     = period_raw.iloc[-1]
    as_of      = period_raw.index[-1].date()

    ret      = ((latest - base) / base * 100).round(2)
    window   = raw[raw.index.date <= end_date].tail(252)
    wk52_h   = window[tickers].max()
    wk52_l   = window[tickers].min()
    log_hl   = np.log(raw_h[tickers] / raw_l[tickers])
    park_vol = (np.sqrt((log_hl**2).mean() / (4 * np.log(2)) * 252) * 100).round(2)

    summary = pd.DataFrame({
        "Ticker":            tickers,
        "Return (%)":        ret[tickers].round(2).values,
        "Current":           latest[tickers].round(2).values,
        "Year Start":        base[tickers].round(2).values,
        "Period High":       period_raw[tickers].max().round(2).values,
        "Period Low":        period_raw[tickers].min().round(2).values,
        "52-Wk High":        wk52_h.round(2).values,
        "52-Wk Low":         wk52_l.round(2).values,
        "Parkinson Vol (%)": park_vol.round(2).values,
    })
    summary["Name"]  = summary["Ticker"].map(etf_names)
    summary["As of"] = as_of

    cap_map   = {style_box[(c, s)]: c for c in CAPS for s in STYLES}
    style_map = {style_box[(c, s)]: s for c in CAPS for s in STYLES}
    summary["Cap"]   = summary["Ticker"].map(cap_map)
    summary["Style"] = summary["Ticker"].map(style_map)

    series = ((period_raw[tickers] - base[tickers]) / base[tickers] * 100).round(4)
    series = series.rename(columns=etf_names)
    series.index = series.index.date

    return summary, series


# ── Style Box HTML ────────────────────────────────────────────────────────────

def render_style_box_html(summary_df, period_label, style_box, etf_names):
    ret_map = summary_df.set_index("Ticker")["Return (%)"].to_dict()
    cur_map = summary_df.set_index("Ticker")["Current"].to_dict()

    col_headers = "".join(
        f'<th style="padding:10px 20px;text-align:center;font-size:14px;'
        f'color:{STYLE_COLORS[s]};font-weight:bold;letter-spacing:0.06em">{s.upper()}</th>'
        for s in STYLES
    )

    rows_html = ""
    for cap in CAPS:
        row_label = (
            f'<td style="padding:10px 14px;font-weight:bold;font-size:13px;'
            f'color:{CAP_COLORS[cap]};white-space:nowrap;vertical-align:middle;'
            f'border-right:2px solid rgba(128,128,128,0.3)">{cap.upper()}</td>'
        )
        cells = ""
        for style in STYLES:
            ticker   = style_box[(cap, style)]
            ret      = ret_map.get(ticker, 0)
            cur      = cur_map.get(ticker, 0)
            color    = "#1a9850" if ret >= 0 else "#d73027"
            is_blend = style == "Blend"
            bg       = "rgba(107,114,128,0.10)" if is_blend else "rgba(255,255,255,0.02)"
            border   = "2px solid rgba(107,114,128,0.4)" if is_blend else "1px solid rgba(128,128,128,0.15)"
            cells += (
                f'<td style="padding:16px 22px;text-align:center;border:{border};'
                f'background:{bg};min-width:150px;vertical-align:middle">'
                f'<div style="font-family:monospace;font-size:16px;font-weight:bold;color:#ccc">{ticker}</div>'
                f'<div style="font-size:11px;color:#888;margin:3px 0 8px 0">{etf_names.get(ticker, "")}</div>'
                f'<div style="font-size:26px;font-weight:bold;color:{color}">{ret:+.2f}%</div>'
                f'<div style="font-size:11px;color:#aaa;margin-top:4px">${cur:.2f}</div>'
                f'</td>'
            )
        rows_html += f"<tr>{row_label}{cells}</tr>"

    return (
        f'<div style="overflow-x:auto">'
        f'<table style="border-collapse:collapse;font-family:sans-serif;margin:8px 0">'
        f'<thead><tr><th style="padding:8px 14px;font-size:11px;color:#666">Period: {period_label}</th>'
        f'{col_headers}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


# ── Returns HTML table with price bars ───────────────────────────────────────

def render_returns_html(df, period_label, preset_label):
    def _fp(v):
        try:
            return f"${float(v):.2f}" if v == v else "—"
        except Exception:
            return "—"

    def _fpct(v, plus=False):
        try:
            f = float(v)
            return "—" if f != f else (f"{f:+.2f}%" if plus else f"{f:.2f}%")
        except Exception:
            return "—"

    def _fratio(v):
        try:
            f = float(v)
            return "—" if f != f else f"{f:.2%}"
        except Exception:
            return "—"

    def _color(v):
        try:
            return "#1a9850" if float(v) >= 0 else "#d73027"
        except Exception:
            return "#888"

    rows = []
    for _, row in df.iterrows():
        ret   = row["Return (%)"]
        color = _color(ret)
        svg   = make_price_bar_svg(row["Period Low"], row["Year Start"], row["Current"], row["Period High"], inline=True)
        hlp   = row.get("Period HL/Price")
        pv    = row.get("Parkinson Vol (%)")
        cap_c = CAP_COLORS.get(row.get("Cap", ""), "#888")
        sty_c = STYLE_COLORS.get(row.get("Style", ""), "#888")
        rows.append(
            f'<tr style="border-bottom:1px solid rgba(128,128,128,0.2)">'
            f'<td style="padding:6px 10px;white-space:nowrap;width:130px">'
            f'  <span style="font-size:11px;font-weight:bold;color:{cap_c};margin-right:4px">{row.get("Cap","")}</span>'
            f'  <span style="font-size:11px;color:{sty_c}">{row.get("Style","")}</span>'
            f'</td>'
            f'<td style="padding:6px 10px;font-family:monospace;font-weight:bold;width:55px">{row["Ticker"]}</td>'
            f'<td style="padding:6px 10px;font-size:12px;color:#999;white-space:nowrap;width:140px">{row["Name"]}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{color};font-weight:bold;white-space:nowrap;width:70px">{ret:+.2f}%</td>'
            f'<td style="padding:6px 10px;width:680px;min-width:680px">{svg}</td>'
            f'<td style="padding:6px 10px;text-align:right;font-family:monospace;white-space:nowrap;width:90px">{_fratio(hlp)}</td>'
            f'<td style="padding:6px 10px;text-align:right;font-family:monospace;white-space:nowrap;width:90px">{_fpct(pv)}</td>'
            f'<td style="padding:6px 10px;text-align:right;font-family:monospace;white-space:nowrap;width:90px">{_fp(row.get("52-Wk Low"))}</td>'
            f'<td style="padding:6px 10px;text-align:right;font-family:monospace;white-space:nowrap;width:90px">{_fp(row.get("52-Wk High"))}</td>'
            f'</tr>'
        )

    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:sans-serif">'
        f'<thead><tr style="border-bottom:2px solid rgba(128,128,128,0.4)">'
        f'<th style="padding:8px 10px;text-align:left;width:130px">Cap / Style</th>'
        f'<th style="padding:8px 10px;text-align:left;width:55px">Ticker</th>'
        f'<th style="padding:8px 10px;text-align:left;width:140px">Name</th>'
        f'<th style="padding:8px 10px;text-align:right;width:70px;white-space:nowrap">Return</th>'
        f'<th style="padding:8px 10px;text-align:left">Price Range &nbsp;&#8212;&nbsp; Low &nbsp;|&nbsp; &#9670; Period-Start &nbsp;|&nbsp; &#9679; Current &nbsp;|&nbsp; High</th>'
        f'<th style="padding:8px 10px;text-align:right;width:90px;white-space:nowrap">HL/Price ({preset_label})</th>'
        f'<th style="padding:8px 10px;text-align:right;width:90px;white-space:nowrap">Parkinson Vol ({preset_label})</th>'
        f'<th style="padding:8px 10px;text-align:right;width:90px;white-space:nowrap">52-Wk Low</th>'
        f'<th style="padding:8px 10px;text-align:right;width:90px;white-space:nowrap">52-Wk High</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

today = date.today()
PRESETS = {
    "YTD":    date(today.year, 1, 1),
    "1Y":     today - timedelta(days=365),
    "2Y":     today - timedelta(days=365 * 2),
    "3Y":     today - timedelta(days=365 * 3),
    "5Y":     today - timedelta(days=365 * 5),
    "Custom": None,
}

if "sb_start_date" not in st.session_state:
    st.session_state.sb_start_date = PRESETS["YTD"]

def on_preset():
    v = PRESETS[st.session_state.sb_preset]
    if v is not None:
        st.session_state.sb_start_date = v

def on_date():
    st.session_state.sb_preset = "Custom"

with st.sidebar:
    st.header("Fund Family")
    family = st.radio("", list(FUND_FAMILIES.keys()), horizontal=True, key="sb_family", label_visibility="collapsed")

    st.header("Date Range")
    start_date = st.date_input("Start Date", max_value=today, key="sb_start_date", on_change=on_date)
    end_date   = st.date_input("End Date", value=today, min_value=start_date, key="sb_end_date", on_change=on_date)
    st.radio("Quick Select", list(PRESETS.keys()), horizontal=True, index=0, key="sb_preset", on_change=on_preset)
    if st.button("Refresh Data", type="primary"):
        st.cache_data.clear()

# ── Resolve active fund family config ─────────────────────────────────────────

style_box = FUND_FAMILIES[family]["style_box"]
etf_names = FUND_FAMILIES[family]["etf_names"]
tickers   = [style_box[(c, s)] for c in CAPS for s in STYLES]

preset_label = st.session_state.get("sb_preset", "Period")
period_label = f"{start_date.strftime('%m-%d-%Y')} --> {end_date.strftime('%m-%d-%Y')}"

st.title("Equity Style Box Dashboard")
st.caption(
    f"**{family}** ETF 9-box grid — Value / Blend / Growth × Large / Mid / Small Cap · "
    f"Period: **{period_label}**"
)

# ── Main ──────────────────────────────────────────────────────────────────────

try:
    summary, series = fetch_data(start_date, end_date, tuple(tickers), family)

    # ── Style Box ─────────────────────────────────────────────────────────────
    st.subheader("Style Box")
    st.markdown(render_style_box_html(summary, period_label, style_box, etf_names), unsafe_allow_html=True)

    # ── Summary metrics: by Cap and by Style ──────────────────────────────────
    st.subheader("Average Returns by Segment")
    col_cap, col_style = st.columns(2)

    with col_cap:
        st.caption("By Cap Tier")
        cap_avgs = summary.groupby("Cap")["Return (%)"].mean().reindex(CAPS).round(2)
        for cap, val in cap_avgs.items():
            color = "#1a9850" if val >= 0 else "#d73027"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;'
                f'border-left:4px solid {CAP_COLORS[cap]};margin-bottom:4px">'
                f'<span style="font-weight:bold">{cap}</span>'
                f'<span style="color:{color};font-weight:bold">{val:+.2f}%</span></div>',
                unsafe_allow_html=True,
            )

    with col_style:
        st.caption("By Style")
        style_avgs = summary.groupby("Style")["Return (%)"].mean().reindex(STYLES).round(2)
        for style, val in style_avgs.items():
            color = "#1a9850" if val >= 0 else "#d73027"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;'
                f'border-left:4px solid {STYLE_COLORS[style]};margin-bottom:4px">'
                f'<span style="font-weight:bold">{style}</span>'
                f'<span style="color:{color};font-weight:bold">{val:+.2f}%</span></div>',
                unsafe_allow_html=True,
            )

    # ── Performance Charts ────────────────────────────────────────────────────
    st.subheader(f"Performance — {period_label}")
    tab_bar, tab_line = st.tabs(["Bar Chart", "Line Chart"])

    with tab_bar:
        color_by  = st.radio("Color by", ["Style", "Cap"], horizontal=True, key="sb_bar_color")
        color_map = STYLE_COLORS if color_by == "Style" else CAP_COLORS
        bar_df    = summary.sort_values("Return (%)", ascending=False)

        fig_bar = px.bar(
            bar_df, x="Ticker", y="Return (%)",
            color=color_by,
            color_discrete_map=color_map,
            text=bar_df["Return (%)"].apply(lambda v: f"{v:+.2f}%"),
            hover_data={"Name": True, "Cap": True, "Style": True, "Return (%)": False, color_by: False},
            title=f"Return by ETF — {period_label}",
            height=480,
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(xaxis_title="", yaxis_ticksuffix="%", coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_line:
        all_names = list(series.columns)
        if "sb_line_sel" not in st.session_state:
            st.session_state.sb_line_sel = {n: True for n in all_names}

        c1, c2, _ = st.columns([1, 1, 8])
        if c1.button("Select All", key="sb_sel_all"):
            for n in all_names:
                st.session_state.sb_line_sel[n] = True
                st.session_state[f"sb_chk_{n}"] = True
        if c2.button("Clear All", key="sb_clr_all"):
            for n in all_names:
                st.session_state.sb_line_sel[n] = False
                st.session_state[f"sb_chk_{n}"] = False

        chk_cols = st.columns(5)
        for i, name in enumerate(all_names):
            st.session_state.sb_line_sel[name] = chk_cols[i % 5].checkbox(
                name, value=st.session_state.sb_line_sel.get(name, True), key=f"sb_chk_{name}"
            )

        selected = [n for n, v in st.session_state.sb_line_sel.items() if v]
        if selected:
            long_df = (
                series[selected].reset_index()
                .rename(columns={"index": "Date"})
                .melt(id_vars="Date", var_name="ETF", value_name="Return (%)")
            )
            fig_line = px.line(long_df, x="Date", y="Return (%)", color="ETF",
                               title=f"Cumulative Return (%) — {period_label}", height=500)
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_line.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Select at least one ETF to display the chart.")

    # ── Factor Analysis ───────────────────────────────────────────────────────
    st.subheader("Factor Analysis")

    # Rename series columns from ETF names back to tickers for spread math
    t_series = series.rename(columns={v: k for k, v in etf_names.items()})

    tab_vg, tab_sz, tab_rot = st.tabs(["Value vs Growth", "Size Premium", "Factor Rotation"])

    with tab_vg:
        st.caption("Value minus Growth return spread by cap tier. Positive = Value outperforming.")
        smooth_vg = st.slider("Smoothing (days)", 1, 20, 1, key="sb_vg_smooth")

        vg_data = {
            f"{cap}: Value − Growth": t_series[style_box[(cap, "Value")]] - t_series[style_box[(cap, "Growth")]]
            for cap in CAPS
        }
        vg_df = pd.DataFrame(vg_data)
        if smooth_vg > 1:
            vg_df = vg_df.rolling(smooth_vg).mean()
        vg_df.index = pd.to_datetime(vg_df.index)

        mc = st.columns(3)
        for i, col_name in enumerate(vg_df.columns):
            latest_v = vg_df[col_name].iloc[-1]
            prev5_v  = vg_df[col_name].iloc[-6] if len(vg_df) > 5 else vg_df[col_name].iloc[0]
            mc[i].metric(col_name, f"{latest_v:+.2f}%", delta=f"{latest_v - prev5_v:+.2f}% (5d)")

        long_vg = (
            vg_df.reset_index().rename(columns={"index": "Date"})
            .melt(id_vars="Date", var_name="Spread", value_name="Return (%)")
        )
        fig_vg = px.line(long_vg, x="Date", y="Return (%)", color="Spread", height=420,
                         title=f"Value − Growth Spread — {period_label}")
        fig_vg.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_vg.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
        st.plotly_chart(fig_vg, use_container_width=True)

    with tab_sz:
        st.caption("Small minus Large return spread by style. Positive = Small-cap outperforming.")
        smooth_sz = st.slider("Smoothing (days)", 1, 20, 1, key="sb_sz_smooth")

        sz_data = {
            f"{style}: Small − Large": t_series[style_box[("Small", style)]] - t_series[style_box[("Large", style)]]
            for style in STYLES
        }
        sz_df = pd.DataFrame(sz_data)
        if smooth_sz > 1:
            sz_df = sz_df.rolling(smooth_sz).mean()
        sz_df.index = pd.to_datetime(sz_df.index)

        mc2 = st.columns(3)
        for i, col_name in enumerate(sz_df.columns):
            latest_v = sz_df[col_name].iloc[-1]
            prev5_v  = sz_df[col_name].iloc[-6] if len(sz_df) > 5 else sz_df[col_name].iloc[0]
            mc2[i].metric(col_name, f"{latest_v:+.2f}%", delta=f"{latest_v - prev5_v:+.2f}% (5d)")

        long_sz = (
            sz_df.reset_index().rename(columns={"index": "Date"})
            .melt(id_vars="Date", var_name="Spread", value_name="Return (%)")
        )
        fig_sz = px.line(long_sz, x="Date", y="Return (%)", color="Spread", height=420,
                         title=f"Small − Large Spread — {period_label}")
        fig_sz.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sz.update_layout(xaxis_title="", yaxis_ticksuffix="%", legend_title="")
        st.plotly_chart(fig_sz, use_container_width=True)

    with tab_rot:
        st.caption(
            "Aggregate factor rotation: avg Value−Growth across all cap tiers and avg Small−Large across all styles. "
            "Green shading = factor outperforming, red = underperforming."
        )

        avg_vg = pd.DataFrame({
            cap: t_series[style_box[(cap, "Value")]] - t_series[style_box[(cap, "Growth")]]
            for cap in CAPS
        }).mean(axis=1)

        avg_sz = pd.DataFrame({
            sty: t_series[style_box[("Small", sty)]] - t_series[style_box[("Large", sty)]]
            for sty in STYLES
        }).mean(axis=1)

        sig_col1, sig_col2 = st.columns(2)
        vg_now = avg_vg.iloc[-1]
        sz_now = avg_sz.iloc[-1]
        sig_col1.metric(
            "Value vs Growth Signal",
            "Value Leading 🟢" if vg_now > 0 else "Growth Leading 🔴",
            delta=f"{vg_now:+.2f}%",
        )
        sig_col2.metric(
            "Size Signal",
            "Small-Cap Leading 🟢" if sz_now > 0 else "Large-Cap Leading 🔴",
            delta=f"{sz_now:+.2f}%",
        )

        fig_rot = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Value − Growth (avg across cap tiers)", "Small − Large (avg across styles)"],
            vertical_spacing=0.10,
        )
        for row, (s_data, line_color) in enumerate([(avg_vg, "#3b82f6"), (avg_sz, "#f59e0b")], start=1):
            dates = pd.to_datetime(s_data.index)
            y     = s_data.values
            fig_rot.add_trace(go.Scatter(x=dates, y=np.where(y > 0, y, 0),
                                         fill="tozeroy", fillcolor="rgba(34,197,94,0.15)",
                                         line=dict(width=0), showlegend=False, hoverinfo="skip"),
                              row=row, col=1)
            fig_rot.add_trace(go.Scatter(x=dates, y=np.where(y < 0, y, 0),
                                         fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                                         line=dict(width=0), showlegend=False, hoverinfo="skip"),
                              row=row, col=1)
            fig_rot.add_trace(go.Scatter(x=dates, y=y, line=dict(color=line_color, width=1.8),
                                         showlegend=False),
                              row=row, col=1)
            fig_rot.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=1)

        fig_rot.update_yaxes(ticksuffix="%")
        fig_rot.update_layout(height=500, xaxis_title="")
        st.plotly_chart(fig_rot, use_container_width=True)

    # ── Returns & Price Table ─────────────────────────────────────────────────
    st.subheader(f"Returns & Price — {period_label}")

    table_df = summary.copy()
    table_df["Period HL/Price"] = (
        (table_df["Period High"] - table_df["Period Low"]) / table_df["Current"]
    ).round(4)
    table_df = table_df.sort_values("Return (%)", ascending=False).reset_index(drop=True)

    st.markdown(render_returns_html(table_df, period_label, preset_label), unsafe_allow_html=True)

    st.caption(
        f"Prices as of {summary['As of'].iloc[0]}. "
        f"Return from {start_date.strftime('%m-%d-%Y')} → {end_date.strftime('%m-%d-%Y')}. "
        f"52-week range = trailing 252 trading days. "
        f"Price bar: gray=period range, ◆=period-start, ●=current."
    )

except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("Please check your internet connection and try refreshing.")
