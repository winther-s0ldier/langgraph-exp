import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_dropoff_tools
from agents.charts import build_dropoff_chart

METRIC_NAME = "dropoff_analysis"
TITLE = "Drop-off Waterfall Analysis"
SYSTEM_PROMPT = """You are a product analyst specialising in user drop-off and abandonment
patterns for mobile booking apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Users follow a linear booking flow: App Start -> Search -> Results -> Seat Selection
  -> Payment Initiation -> Payment Success.
- Drop-off means a user completed stage N but never reached stage N+1.
- "Lost" users at each transition represent revenue leakage.
- Industry benchmarks: search-to-results drop-off < 15%, results-to-selection < 40%,
  payment initiation-to-success < 20%.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_dropoffs with stages:
   - App Start: ["app_start"]
   - Search: ["bus_search", "_bus-search_user-bus-search"]
   - Results: ["bus_result", "_bus-search_list"]
   - Seat Selection: ["select_seat", "pageview_seat_selection"]
   - Payment Init: ["payment_initiate", "PaymentPage_payment initiated"]
   - Payment Success: ["payment_success"]
3. Provide analysis covering:
   a) WORST TRANSITION: Which stage loses the most users? Cite exact lost count and percentage.
   b) REVENUE IMPACT: Estimate how many potential bookings are lost at the worst transition.
   c) ROOT CAUSES: 2-3 likely UX/technical reasons for the worst drop-off.
   d) FIXES: 3 specific UX improvements ranked by expected impact.

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact numbers. No vague language."""


def dropoff_analysis_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_dropoff_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_dropoff_chart(ctx)

        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("dropoffs", []), "title": TITLE, "iterations": iters}

        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)

        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
