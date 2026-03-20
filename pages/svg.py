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

st.set_page_config(page_title="SVG price bar", layout="wide")


# ── SVG price-bar helpers ──────────────────────────────────────────────────────

def make_price_bar_svg(ytd_low, year_start, current, ytd_high, width=540, inline=False):
    """
    Horizontal range bar — height is auto-calculated so all labels are always visible.
      Above bar: year_start (gray) and current (colored, bold) — collision-aware
      Below bar: period low (left) and period high (right)
      On bar:    colored fill from year_start→current, diamond at year_start, dot at current
    """
    pad_l, pad_r = 40, 40
    font_size    = 7.5
    top_pad      = 2
    above_y      = top_pad + font_size          # label baseline above bar
    bar_gap      = 4                            # gap: label baseline → bar top
    bar_y        = above_y + bar_gap            # ~13.5
    bar_h        = 2
    below_gap    = 7                            # gap: bar bottom → below label baseline
    below_y      = bar_y + bar_h + below_gap    # ~22.5
    bottom_pad   = 2
    height       = int(below_y + bottom_pad)    # auto height (~24px)

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

    color   = "#1a9850" if current >= year_start else "#d73027"
    fill_x  = min(x_start, x_curr)
    fill_w  = max(abs(x_curr - x_start), 1)
    bar_mid = bar_y + bar_h / 2

    # ── above-bar label collision avoidance ────────────────────────────────────
    min_gap = 46
    raw_sx, raw_cx = x_start, x_curr
    if abs(raw_cx - raw_sx) < min_gap:
        mid = (raw_sx + raw_cx) / 2
        half = min_gap / 2
        raw_sx, raw_cx = (mid - half, mid + half) if x_start <= x_curr else (mid + half, mid - half)
    start_lbl_x = max(pad_l + 12, min(width - pad_r - 12, raw_sx))
    curr_lbl_x  = max(pad_l + 12, min(width - pad_r - 12, raw_cx))

    # ── diamond marker ─────────────────────────────────────────────────────────
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
  <rect x="{x_lo:.1f}" y="{bar_y:.1f}" width="{x_hi-x_lo:.1f}" height="{bar_h}" rx="2" fill="#d9d9d9"/>
  <!-- Colored fill (period-start → current) -->
  <rect x="{fill_x:.1f}" y="{bar_y:.1f}" width="{fill_w:.1f}" height="{bar_h}" fill="{color}" opacity="0.85" rx="1"/>
  <!-- Period-start dashed line -->
  <line x1="{x_start:.1f}" y1="{bar_y-4:.1f}" x2="{x_start:.1f}" y2="{bar_y+bar_h:.1f}"
        stroke="#555" stroke-width="1" stroke-dasharray="2,1.5"/>
  <!-- Period-start diamond -->
  <polygon points="{diamond}" fill="#555"/>
  <!-- Low end-cap -->
  <line x1="{x_lo:.1f}" y1="{bar_y-1:.1f}" x2="{x_lo:.1f}" y2="{bar_y+bar_h+1:.1f}" stroke="#999" stroke-width="1"/>
  <!-- High end-cap -->
  <line x1="{x_hi:.1f}" y1="{bar_y-1:.1f}" x2="{x_hi:.1f}" y2="{bar_y+bar_h+1:.1f}" stroke="#999" stroke-width="1"/>
  <!-- Current price dot -->
  <circle cx="{x_curr:.1f}" cy="{bar_mid:.1f}" r="3" fill="{color}" stroke="white" stroke-width="1"/>
  <!-- Period-start label (above bar, gray) -->
  <text x="{start_lbl_x:.1f}" y="{above_y:.1f}" font-size="{font_size}" fill="#555"
        font-family="sans-serif" text-anchor="middle">${year_start:.2f}</text>
  <!-- Current price label (above bar, bold, colored) -->
  <text x="{curr_lbl_x:.1f}" y="{above_y:.1f}" font-size="{font_size}" fill="{color}"
        font-weight="bold" font-family="sans-serif" text-anchor="middle">${current:.2f}</text>
  <!-- Period Low label (below bar, left) -->
  <text x="{x_lo:.1f}" y="{below_y:.1f}" font-size="{font_size}" fill="#888"
        font-family="sans-serif" text-anchor="start">${ytd_low:.2f}</text>
  <!-- Period High label (below bar, right) -->
  <text x="{x_hi:.1f}" y="{below_y:.1f}" font-size="{font_size}" fill="#888"
        font-family="sans-serif" text-anchor="end">${ytd_high:.2f}</text>
</svg>"""
    return svg


def svg_to_data_url(svg_str):
    encoded = base64.b64encode(svg_str.encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"


def render_ytd_html(ytd_df):
    rows = []
    for _, row in ytd_df.iterrows():
        ytd_val = row["Return (%)"]
        color   = "#1a9850" if ytd_val >= 0 else "#d73027"
        svg     = make_price_bar_svg(
            row["Period Low"], row["Year Start"], row["Current"], row["Period High"],
            inline=True,
        )
        is_benchmark = row["Ticker"] in ("SPY", "RSP")
        bg        = ""
        txt_style = "font-weight:bold;font-size:15px;" if is_benchmark else ""
        park = row.get("Parkinson Vol (%)", float("nan"))
        park_str = f"{park:.2f}%" if park == park else "—"
        phl = row.get("HL/Price (%)")
        phl_str = f"{phl:.2f}%" if phl is not None else "—"
        rows.append(f"""
        <tr style="border-bottom:1px solid rgba(128,128,128,0.2);{bg}">
          <td style="padding:6px 12px;white-space:nowrap;{txt_style}">{row['Name']}</td>
          <td style="padding:6px 12px;font-family:monospace;{txt_style}">{row['Ticker']}</td>
          <td style="padding:6px 12px;text-align:right;color:{color};font-weight:bold;white-space:nowrap;{txt_style}">
            {ytd_val:+.2f}%</td>
          <td style="padding:6px 12px;text-align:right;white-space:nowrap;{txt_style}">{park_str}</td>
          <td style="padding:6px 12px;text-align:right;white-space:nowrap;{txt_style}">{phl_str}</td>
          <td style="padding:6px 12px">{svg}</td>
        </tr>""")

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:sans-serif">
      <thead>
        <tr style="border-bottom:2px solid rgba(128,128,128,0.4)">
          <th style="padding:8px 12px;text-align:left;white-space:nowrap">Name</th>
          <th style="padding:8px 12px;text-align:left">Ticker</th>
          <th style="padding:8px 12px;text-align:right;white-space:nowrap">Return</th>
          <th style="padding:8px 12px;text-align:right;white-space:nowrap">Park Vol</th>
          <th style="padding:8px 12px;text-align:right;white-space:nowrap">HL/Price</th>
          <th style="padding:8px 12px;text-align:left">
            Price Range &nbsp;&#8212;&nbsp; Low &nbsp;|&nbsp; &#9670; Period-Start &nbsp;|&nbsp; &#9679; Current &nbsp;|&nbsp; High
          </th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def fetch_all_data(tickers, start_date: date, end_date: date):
    ohlcv = yf.download(tickers, start=start_date, end=end_date, progress=False)
    return ohlcv["Close"], ohlcv["High"], ohlcv["Low"]


#### main page logic #############################################################################
if __name__ == "__main__":
    st.title("SVG price bar")

    end_date = date.today()
    #start_date = end_date - timedelta(days=365)
    start_date = date(end_date.year, 1, 1)  # calendar year to date

    tickers = {
        "SPY": "S&P 500",
        "RSP": "S&P 500 Equal Weight",
        "XLC": "Communication Services",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLK": "Technology",
        "XLB": "Materials",
        "XLF": "Financials",
        'XLRE': "Real Estate",
        "XLY": "Consumer Discretionary",
        'XLP': "Consumer Staples",
        "XLU": "Utilities",
        'XLV': "Healthcare",
    }

    data, data_h, data_l = fetch_all_data(list(tickers.keys()), start_date, end_date)
    st.text(f"Data from {start_date} to {end_date}")

    ytd_data = []
    for ticker, name in tickers.items():
        series = data[ticker].dropna()
        if len(series) < 2:
            continue
        year_start = series.iloc[0]
        current    = series.iloc[-1]
        ytd_low    = series.min()
        ytd_high   = series.max()
        ytd_return = (current / year_start - 1) * 100

        # Parkinson volatility (annualised %)
        h = data_h[ticker].dropna()
        l = data_l[ticker].dropna()
        lhl      = np.log(h / l)
        park_vol = round(float(np.sqrt((lhl ** 2).mean() / (4 * np.log(2)) * 252) * 100), 2)

        hl_range = ytd_high - ytd_low
        price_hl = round(float(hl_range / current * 100), 2) if current else None

        ytd_data.append({
            "Ticker":            ticker,
            "Name":              name,
            "Year Start":        year_start,
            "Current":           current,
            "Period Low":        ytd_low,
            "Period High":       ytd_high,
            "Return (%)":        ytd_return,
            "Parkinson Vol (%)": park_vol,
            "HL/Price (%)":      price_hl,
        })

    ytd_df = pd.DataFrame(ytd_data).sort_values("Return (%)", ascending=False)
    html = render_ytd_html(ytd_df)
    st.markdown(html, unsafe_allow_html=True)