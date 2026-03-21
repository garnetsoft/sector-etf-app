import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import calendar

st.set_page_config(page_title="DOW 30 Financial Calendar", layout="wide")

DOW_30 = {
    "AAPL": "Apple",
    "AMGN": "Amgen",
    "AXP": "American Express",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "DIS": "Disney",
    "DOW": "Dow Inc.",
    "GS": "Goldman Sachs",
    "HD": "Home Depot",
    "HON": "Honeywell",
    "IBM": "IBM",
    "INTC": "Intel",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "MCD": "McDonald's",
    "MMM": "3M",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "NKE": "Nike",
    "PG": "Procter & Gamble",
    "SHW": "Sherwin-Williams",
    "TRV": "Travelers",
    "UNH": "UnitedHealth",
    "V": "Visa",
    "VZ": "Verizon",
    "WMT": "Walmart",
}

EVENT_COLORS = {
    "Earnings": "#4C9BE8",
    "Ex-Dividend": "#F4A261",
    "Dividend Payment": "#2A9D8F",
    "Split": "#E76F51",
}

st.title("DOW 30 Financial Calendar")
st.caption("Upcoming earnings, dividends, and corporate actions for Dow Jones Industrial Average components.")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_cols = st.columns(len(EVENT_COLORS))
for col, (etype, color) in zip(legend_cols, EVENT_COLORS.items()):
    col.markdown(
        f'<span style="background:{color};border-radius:4px;padding:3px 10px;color:white;font-size:0.85em">{etype}</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Controls ───────────────────────────────────────────────────────────────────
lookahead_map = {"1M": 30, "2M": 60, "3M": 90, "6M": 180, "1Y": 365}
lookahead_choice = st.radio("Lookahead", list(lookahead_map.keys()), horizontal=True)
lookahead_days = lookahead_map[lookahead_choice]
selected_tickers = st.multiselect(
    "Companies",
    options=list(DOW_30.keys()),
    default=list(DOW_30.keys()),
)
selected_event_types = st.multiselect(
    "Event Types",
    options=list(EVENT_COLORS.keys()),
    default=list(EVENT_COLORS.keys()),
)
view_mode = st.radio("View", ["Calendar", "Table"], horizontal=True)

# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_events(tickers: tuple, lookahead: int) -> pd.DataFrame:
    today = date.today()
    raw_end = today + timedelta(days=lookahead)
    end_date = raw_end.replace(day=calendar.monthrange(raw_end.year, raw_end.month)[1])
    rows = []

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)

            # --- Earnings (from earnings_dates — future rows have NaN Reported EPS) ---
            try:
                ed_df = tk.earnings_dates
                if ed_df is not None and not ed_df.empty:
                    future = ed_df[ed_df["Reported EPS"].isna()]
                    for idx in future.index:
                        try:
                            ed = pd.to_datetime(idx).date()
                            if today <= ed <= end_date:
                                rows.append({
                                    "Date": ed,
                                    "Ticker": ticker,
                                    "Company": DOW_30[ticker],
                                    "Event": "Earnings",
                                    "Detail": f"Est. EPS: {future.loc[idx, 'EPS Estimate']:.2f}"
                                              if pd.notna(future.loc[idx, "EPS Estimate"]) else "",
                                })
                        except Exception:
                            pass
            except Exception:
                pass

            # --- Calendar dict: ex-dividend and dividend payment dates ---
            try:
                cal = tk.calendar
                if isinstance(cal, dict):
                    # Ex-dividend date
                    ex_div = cal.get("Ex-Dividend Date")
                    if ex_div:
                        ex_div = pd.to_datetime(ex_div).date()
                        if today <= ex_div <= end_date:
                            # get latest dividend amount from history
                            try:
                                divs = tk.dividends
                                amount = divs.iloc[-1] if not divs.empty else None
                            except Exception:
                                amount = None
                            rows.append({
                                "Date": ex_div,
                                "Ticker": ticker,
                                "Company": DOW_30[ticker],
                                "Event": "Ex-Dividend",
                                "Detail": f"${amount:.4f}" if amount else "",
                            })

                    # Dividend payment date
                    div_date = cal.get("Dividend Date")
                    if div_date:
                        div_date = pd.to_datetime(div_date).date()
                        if today <= div_date <= end_date:
                            try:
                                divs = tk.dividends
                                amount = divs.iloc[-1] if not divs.empty else None
                            except Exception:
                                amount = None
                            rows.append({
                                "Date": div_date,
                                "Ticker": ticker,
                                "Company": DOW_30[ticker],
                                "Event": "Dividend Payment",
                                "Detail": f"${amount:.4f}" if amount else "",
                            })
            except Exception:
                pass

            # --- Stock splits from actions ---
            try:
                actions = tk.actions
                if actions is not None and not actions.empty:
                    # Normalize timezone-aware index to plain dates
                    dates = actions.index.tz_localize(None) if actions.index.tzinfo is None else actions.index.tz_convert(None)
                    actions = actions.copy()
                    actions.index = pd.DatetimeIndex(dates).normalize()
                    future_actions = actions[
                        (actions.index.date >= today) & (actions.index.date <= end_date)
                    ]
                    for idx, row in future_actions.iterrows():
                        split_val = row.get("Stock Splits", 0)
                        if split_val not in (0, 1):
                            rows.append({
                                "Date": idx.date(),
                                "Ticker": ticker,
                                "Company": DOW_30[ticker],
                                "Event": "Split",
                                "Detail": f"{split_val:.2f}:1",
                            })
            except Exception:
                pass

        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "Company", "Event", "Detail"])

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates().sort_values("Date")
    return df


# ── Load data ─────────────────────────────────────────────────────────────────
if not selected_tickers:
    st.warning("Select at least one company above.")
    st.stop()

with st.spinner("Fetching upcoming events…"):
    df_all = fetch_events(tuple(sorted(selected_tickers)), lookahead_days)

# Apply event-type filter
df = df_all[df_all["Event"].isin(selected_event_types)].copy() if not df_all.empty else df_all

today = date.today()
raw_end = today + timedelta(days=lookahead_days)
end_date = raw_end.replace(day=calendar.monthrange(raw_end.year, raw_end.month)[1])

# ── Summary metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
counts = df["Event"].value_counts() if not df.empty else {}
col1.metric("Earnings", counts.get("Earnings", 0))
col2.metric("Ex-Dividend", counts.get("Ex-Dividend", 0))
col3.metric("Dividend Payment", counts.get("Dividend Payment", 0))
col4.metric("Splits", counts.get("Split", 0))

st.markdown("---")

if df.empty:
    st.info("No upcoming events found for the selected filters and time range.")
    st.stop()

# ── Views ─────────────────────────────────────────────────────────────────────

def color_event(val):
    color = EVENT_COLORS.get(val, "#888")
    return f"background-color:{color}22;color:{color};font-weight:600"


if view_mode == "Table":
    display = df.copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Days Away"] = (
        pd.to_datetime(display["Date"]) - pd.Timestamp(today)
    ).dt.days

    # Group by week
    st.subheader(f"Events: {today.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}")

    styled = (
        display[["Date", "Days Away", "Ticker", "Company", "Event", "Detail"]]
        .reset_index(drop=True)
        .style.applymap(color_event, subset=["Event"])
        .format({"Days Away": lambda x: f"+{x}d"})
    )
    st.dataframe(styled, use_container_width=True, height=600)

else:  # Calendar view
    # Build month grids for all months in range
    months_in_range = sorted(
        set((df["Date"].dt.year * 100 + df["Date"].dt.month).unique())
    )

    # Build event lookup: date -> list of (ticker, event)
    event_map: dict[date, list] = {}
    for _, row in df.iterrows():
        d = row["Date"].date()
        event_map.setdefault(d, []).append((row["Ticker"], row["Event"]))

    def render_month_html(year: int, month: int) -> str:
        """Return a self-contained HTML table for one month."""
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        header_cells = "".join(
            f'<th style="text-align:center;padding:4px 2px;font-size:0.8em;color:#888">{d}</th>'
            for d in day_names
        )
        rows_html = ""
        for week in calendar.monthcalendar(year, month):
            cells = ""
            for day_num in week:
                if day_num == 0:
                    cells += '<td style="padding:4px 2px;vertical-align:top;min-width:60px"></td>'
                else:
                    d = date(year, month, day_num)
                    is_today = d == today
                    num_style = (
                        "font-weight:700;color:#4C9BE8" if is_today else "font-weight:700;color:#ddd"
                    )
                    chips = ""
                    for ticker, etype in event_map.get(d, []):
                        color = EVENT_COLORS.get(etype, "#888")
                        chips += (
                            f'<div style="background:{color};border-radius:3px;padding:1px 4px;'
                            f'color:white;font-size:0.68em;margin-top:2px;white-space:nowrap;overflow:hidden;">'
                            f'{ticker}</div>'
                        )
                    cells += (
                        f'<td style="padding:4px 2px;vertical-align:top;min-width:60px">'
                        f'<span style="{num_style};font-size:0.85em">{day_num}</span>'
                        f'{chips}</td>'
                    )
            rows_html += f"<tr>{cells}</tr>"

        month_label = f"{calendar.month_name[month]} {year}"
        return (
            f'<div style="margin-bottom:8px;padding:0 16px;">'
            f'<div style="font-weight:700;font-size:1.25em;margin-bottom:8px;color:#fff;'
            f'background:#2a2a3d;border-radius:6px;padding:6px 12px;letter-spacing:0.05em;">{month_label}</div>'
            f'<table style="border-collapse:collapse;width:100%">'
            f'<thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>'
        )

    MONTHS_PER_ROW = 3
    for i in range(0, len(months_in_range), MONTHS_PER_ROW):
        chunk = months_in_range[i : i + MONTHS_PER_ROW]
        cols = st.columns(len(chunk))
        for col, ym in zip(cols, chunk):
            col.markdown(render_month_html(ym // 100, ym % 100), unsafe_allow_html=True)
