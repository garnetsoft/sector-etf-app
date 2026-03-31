import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Shared constants ───────────────────────────────────────────────────────────

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

LOCAL_BASE_URL = "http://localhost:8080/v1"

# USD per 1M tokens: (input_price, output_price)
MODEL_PRICING = {
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-6":           (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


@st.cache_data(ttl=60, show_spinner=False)
def get_local_models() -> dict:
    """Query local llama-server for available models. Returns {display_name: model_id}."""
    try:
        resp = requests.get(f"{LOCAL_BASE_URL}/models", timeout=2)
        resp.raise_for_status()
        result = {}
        for m in resp.json().get("data", []):
            model_id = m["id"]
            display  = model_id.removesuffix(".gguf") + " (Local, Free)"
            result[display] = model_id
        return result
    except Exception:
        return {}


# ── Cost helper ────────────────────────────────────────────────────────────────

def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> str:
    inp_p, out_p = MODEL_PRICING.get(model_id, (0, 0))
    cost = (input_tokens * inp_p + output_tokens * out_p) / 1_000_000
    return f"~${cost:.4f}"


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


# ── S&P 500 constituents ───────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_constituents():
    gics_to_etf = {v: k for k, v in ETF_TO_GICS.items()}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["Ticker", "Name", "GICS Sector", "Sub-Industry"]
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["ETF"]    = df["GICS Sector"].map(gics_to_etf)
    return df.dropna(subset=["ETF"]).reset_index(drop=True)
