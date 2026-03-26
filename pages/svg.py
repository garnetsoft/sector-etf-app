import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

from utils import make_price_bar_svg, svg_to_data_url

st.set_page_config(page_title="SVG price bar", layout="wide")


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