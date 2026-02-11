import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_retention_tools
from agents.charts import build_retention_chart

METRIC_NAME = "retention_analysis"
TITLE = "Cohort Retention Heatmap"
SYSTEM_PROMPT = """You are a retention and growth analyst specialising in mobile app
lifecycle analysis. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Cohort retention measures what percentage of users from a given week return in subsequent weeks.
- Week 0 is always 100% (the cohort's first activity). Week 1 retention is the most critical
  metric — it shows how many users come back after their first experience.
- For travel/booking apps, W1 retention of 20-30% is typical. Above 30% is strong.
  W4 retention above 10% indicates a sticky product.
- Cohort labels include the cohort size (n=X). Small cohorts (< 10 users) may show noisy data.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_retention with max_weeks=5 to generate the cohort retention matrix.
3. Provide exactly 4 insights:
   a) BEST COHORT: Which weekly cohort has the strongest retention? Why might that be?
      (Consider if seasonality, marketing campaigns, or product changes could explain it.)
   b) WORST COHORT: Which cohort drops off fastest? Cite exact W1 and W2 percentages.
   c) OVERALL TREND: Is retention improving or declining across successive cohorts?
      This indicates whether product changes are helping or hurting.
   d) RETENTION STRATEGIES: 2 specific, data-backed strategies to improve W1 retention.
      Each strategy should reference a specific finding from the data.

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact retention percentages from the matrix. No vague language."""


def retention_analysis_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_retention_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_retention_chart(ctx)
        data = ctx.get("retention", {})
        serializable = {"w1_pct": data.get("w1_pct"), "labels": data.get("labels"),
                        "max_weeks": data.get("max_weeks")}
        result = {"insights": insights, "fig": chart_html, "data": serializable, "title": TITLE, "iterations": iters}
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": serializable,
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
