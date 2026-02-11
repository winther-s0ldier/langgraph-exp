import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_frequency_tools
from agents.charts import build_frequency_chart

METRIC_NAME = "event_frequency"
TITLE = "Event Frequency Distribution"
SYSTEM_PROMPT = """You are a data analyst specialising in event telemetry and product
instrumentation for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Event frequency distribution reveals which features users interact with most, and which
  are underused or potentially broken.
- A healthy event distribution has a long tail: a few high-frequency events (app_start,
  bus_search) and many lower-frequency feature events.
- If a single event dominates > 30% of all events, it may indicate event spam or a logging bug.
- Power users (P90+ event count) reveal engagement ceiling and product stickiness.
- Events with very low counts (< 10) may be deprecated or broken features.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_frequency_distribution with top_n=20 to get the full distribution.
3. Provide analysis covering:
   a) DOMINANT PATTERNS: Which events dominate? What does this tell us about primary
      user behaviour? Cite exact counts and percentages.
   b) EVENT NOISE: Are there any events that appear suspiciously often (potential logging
      bugs) or suspiciously rarely (broken features)?
   c) POWER USERS: What separates power users (P90+) from casual users? Cite the median
      vs P90 vs P99 event counts.
   d) CONSOLIDATION: Are there duplicate or redundant event names that should be
      consolidated? (e.g., multiple search-related events.)

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact numbers. No vague language."""


def event_frequency_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_frequency_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_frequency_chart(ctx)
        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("frequency_data", {}), "title": TITLE, "iterations": iters}
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
