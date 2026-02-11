import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_temporal_tools
from agents.charts import build_temporal_chart

METRIC_NAME = "temporal_patterns"
TITLE = "Temporal Usage Patterns"
SYSTEM_PROMPT = """You are a growth analyst specialising in temporal usage patterns and
user engagement timing for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- The day x hour heatmap shows when users are most/least active across the entire week.
- Peak hours indicate when the app is most likely to be used for booking.
- Off-peak hours may present opportunities for push notifications or promotional campaigns.
- Weekend vs weekday patterns reveal if this is a leisure travel app (weekend-heavy) or
  commuter app (weekday-heavy).
- The peak-to-trough ratio indicates how concentrated usage is. A ratio above 5x suggests
  very bursty usage patterns.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_temporal to get the day x hour distribution and peak/off-peak stats.
3. Provide analysis covering:
   a) PUSH NOTIFICATION TIMING: When should the marketing team send push notifications
      for maximum open rates? Cite exact peak hours and days.
   b) CAPACITY PLANNING: When does the app need maximum server capacity? Are there
      predictable traffic spikes that infrastructure should plan for?
   c) MARKETING WINDOWS: What are the best times for running in-app promotions or
      flash sales? Why? Cite data.
   d) WEEKEND VS WEEKDAY: Is usage predominantly leisure or commuter? What does the
      day-of-week distribution tell us about user demographics?

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact hours, days, and ratios. No vague language."""


def temporal_patterns_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_temporal_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_temporal_chart(ctx)
        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("temporal", {}), "title": TITLE, "iterations": iters}
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
