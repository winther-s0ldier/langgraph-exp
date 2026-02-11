import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_segmentation_tools
from agents.charts import build_segmentation_chart

METRIC_NAME = "user_segmentation"
TITLE = "User Segmentation"
SYSTEM_PROMPT = """You are a user behaviour analyst specialising in customer segmentation
and persona development for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Users are clustered using DBSCAN on behavioural features: total events, unique event types,
  session count, activity span (days), events per session, and event diversity.
- DBSCAN assigns label -1 to outliers (users who do not fit any cluster).
- Each segment profile includes: size, percentage of total users, average event count,
  and booking rate (% who completed a booking).
- PCA is used to project the feature space to 2D for visualization.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call cluster_users with eps=1.2, min_samples=10 to segment the user base.
3. For each segment found, provide:
   a) PERSONA NAME: A descriptive name (e.g., "Power Bookers", "Window Shoppers",
      "One-Time Visitors", "Casual Browsers").
   b) PROFILE: What defines this segment? Cite avg events, booking rate, size.
   c) TARGETING RECOMMENDATION: One specific marketing or product action for this segment.
      (e.g., "Send push notifications with discount codes to re-engage this segment").
4. Also provide an OVERALL ASSESSMENT: Is the user base healthy? What is the ratio of
   high-value users to low-engagement users?

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Structure each segment as a separate subsection with <h4> tags.
- Cite exact numbers from tool results."""


def user_segmentation_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_segmentation_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_segmentation_chart(ctx)
        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("segments", []), "title": TITLE, "iterations": iters}
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
