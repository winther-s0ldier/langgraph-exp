import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_latency_tools
from agents.charts import build_latency_chart

METRIC_NAME = "time_to_action"
TITLE = "Time-to-Action Analysis"
SYSTEM_PROMPT = """You are a performance analyst specialising in user flow latency and
time-to-action metrics for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Time-to-action measures how long (in seconds) users take to move between two key events.
- For each event pair, we compute: median latency, mean latency, and P90 (90th percentile).
- Transitions capped at 1 hour to exclude users who left and returned days later.
- Industry benchmarks for booking apps:
  * Search to results: < 5 seconds (server+render time)
  * Results to seat selection: 30-120 seconds (browsing time)
  * Seat to payment: 60-180 seconds (seat selection + passenger details)
  * OTP generate to verify: < 60 seconds (should be near-instant)
  * End-to-end (search to payment): 3-8 minutes for a healthy flow.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_latency for each of these event pairs:
   - from_event="bus_search", to_event="bus_result"
   - from_event="bus_result", to_event="select_seat"
   - from_event="select_seat", to_event="payment_initiate"
   - from_event="generate_otp", to_event="verify_otp"
   - from_event="bus_search", to_event="payment_success"
3. Provide analysis covering:
   a) SLOWEST TRANSITION: Which pair has the highest median latency? Is it within benchmarks?
   b) BOTTLENECK ANALYSIS: Is there a technical bottleneck (high P90 vs median gap) or a UX
      issue (high median overall)? Cite exact numbers.
   c) END-TO-END TIME: How long does the full search-to-payment flow take? Is this competitive?
   d) SPEED IMPROVEMENTS: 3 specific ways to reduce latency, each with expected time savings.

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact latency numbers in seconds. No vague language."""


def time_to_action_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_latency_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools, max_iterations=8)
        chart_html = build_latency_chart(ctx)
        pairs = ctx.get("latency_pairs", [])
        serializable = [{"from": p["from"], "to": p["to"], "median": p["median"],
                         "mean": p["mean"], "p90": p["p90"], "n": p["n"]} for p in pairs]
        result = {"insights": insights, "fig": chart_html,
                  "data": serializable, "title": TITLE, "iterations": iters}
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": serializable,
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
