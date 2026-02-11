import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_user_journey_tools
from agents.charts import build_user_journey_chart

METRIC_NAME = "user_journey_insights"
TITLE = "User Journey & Persona Analysis"
SYSTEM_PROMPT = """You are a user journey analyst specialising in behaviour profiling
and persona identification for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- We track individual user journeys to understand how different segments interact with the app.
- Key user milestones: Search -> Select Seat -> Payment Init -> Payment Success.
- User Types:
  * Browsers: Searched but never selected a seat.
  * Shoppers: Selected a seat but never initiated payment.
  * Attempters: Initiated payment but failed/dropped off.
  * Bookers: Successfully completed a booking.
- Friction markers: Users who faced payment failures or app errors during their journey.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_user_journey_stats with min_events=5 to get aggregated journey data.
3. Provide analysis covering:
   a) USER TYPE BREAKDOWN: What % of users are Browsers vs Shoppers vs Bookers?
      What does this funnel shape tell us about intent?
   b) FRICTION IMPACT: How many users faced payment failures or errors?
      Is this a significant churn driver?
   c) POWER USERS VS CASUALS: Based on the stats, what defines a "high-intent" user?
   d) PERSONALIZATION: Suggest 2 ways to tailor the experience for "Browsers" to convert
      them into "Shoppers".

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact counts and percentages from the stats. No vague language."""


def user_journey_insights_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        # Filter to application events only
        df = df[df["category"] == "application"]
        
        ctx = {}
        tools = create_query_tools(df, ctx) + create_user_journey_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_user_journey_chart(ctx)
        
        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("user_journey_stats", {}), "title": TITLE, "iterations": iters}
        
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)
        
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
