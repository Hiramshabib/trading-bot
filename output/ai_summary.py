import os

import pandas as pd


def generate_ai_summary(
    results: pd.DataFrame,
    det_df: pd.DataFrame | None,
    over_df: pd.DataFrame | None,
) -> str:
    """Call Claude to generate a plain-English market summary. Returns empty string if unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    try:
        import anthropic
    except ImportError:
        return ""

    top = results.head(20)

    lines = ["TOP 20 PICKS (ticker | composite | momentum | analyst | fundamentals | 1mo% | RSI):"]
    for _, r in top.iterrows():
        lines.append(
            f"  {r['ticker']:6s} | {r['composite_score']:5.1f} | {r['momentum_score']:5.1f} | "
            f"{r['analyst_score']:5.1f} | {r['fundamentals_score']:5.1f} | "
            f"{r.get('pct_1mo', 0)*100:+.1f}% | RSI {r.get('rsi_14', 50):.0f}"
        )

    if det_df is not None and not det_df.empty:
        high = det_df[det_df["severity"] == "HIGH"]
        if not high.empty:
            lines.append("\nHIGH-SEVERITY DETERIORATION ALERTS: " +
                         ", ".join(high["ticker"].tolist()))

    if over_df is not None and not over_df.empty:
        lines.append("\nOVEREXTENSION WARNINGS (RSI>75): " +
                     ", ".join(over_df["ticker"].tolist()))

    data_summary = "\n".join(lines)

    prompt = f"""You are a concise equity analyst. Based on today's quantitative screening data below, write a 2–3 paragraph plain-English summary for a retail investor. Cover:
1. The top picks and what's driving their scores (momentum, analyst sentiment, or fundamentals).
2. Any notable risks — overbought signals or deterioration alerts.
3. One sentence on the overall market tone implied by the data.

Keep it under 200 words. Do not recommend buying or selling specific stocks.

{data_summary}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
    except Exception as e:
        print(f"  AI summary skipped: {e}")
        return ""
