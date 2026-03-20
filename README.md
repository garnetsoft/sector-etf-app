
❯ how do I view the current git repo from a browser?

● You'll need to push it to a hosting service. The easiest is GitHub:

  1. Create a new repo on GitHub
  Go to https://github.com/new, name it (e.g. sector-etf-app), leave it empty (no README), and click Create repository.

  2. Push your local repo
  git remote add origin https://github.com/garnetsoft/sector-etf-app.git
  git branch -M main
  git push -u origin main

  Then you can view it at https://github.com/garnetsoft/sector-etf-app.

  ---
  Want to also make the app itself viewable in a browser by others? The easiest option is Streamlit Community Cloud (free):
  1. Push the repo to GitHub (above)
  2. Go to https://share.streamlit.io
  3. Connect your GitHub account and deploy — it gives you a public URL like https://garnetsoft-sector-etf-app.streamlit.app


## actually works, amazing -- LAUNCHING VSCODE FROM WSL DIRECTLY!!
Ubuntu -> /home/gfeng/git/sector_etf_app/code . 

https://calendar-component.streamlit.app/



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
