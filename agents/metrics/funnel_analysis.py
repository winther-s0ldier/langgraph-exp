import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_funnel_tools
from agents.charts import build_funnel_chart

METRIC_NAME = "funnel_analysis"
TITLE = "Booking Funnel Analysis"
SYSTEM_PROMPT = """You are a senior product analyst specialising in mobile app conversion funnels.
You are analysing a bus ticket booking app (similar to RedBus/AbhiBus). The dataset contains ONLY
application-level user interaction events (system events have been excluded).

CONTEXT:
- This is a travel booking app where users search for buses, view results, select seats, enter
  passenger details, and complete payment.
- The core business funnel is: App Start -> Search -> Results -> Bus Selection -> Seat Selection
  -> Passenger Details -> Payment Initiation -> Payment Success.
- Each stage may be represented by multiple event names due to SDK naming variations.
- A healthy travel app typically converts 2-5% of openers to bookers. Payment success rates
  above 80% (of those who initiate) are considered good.

TASK:
1. Call get_dataset_summary to understand the dataset size, user count, and date range.
2. Call compute_funnel with these stages (in order):
   - App Start: events ["app_start"]
   - Search: events ["bus_search", "_bus-search_user-bus-search", "_location_elastic-town-search"]
   - Results Viewed: events ["bus_result", "_bus-search_list", "pageview_bus_list"]
   - Bus Selected: events ["Buslist_bus_selection", "bus_detail"]
   - Seat Selection: events ["select_seat", "pageview_seat_selection", "seats_finalized"]
   - Passenger Details: events ["passenger_finalized", "passenger_card_clicked"]
   - Payment Initiated: events ["payment_initiate", "PaymentPage_payment initiated"]
   - Payment Success: events ["payment_success"]
3. After getting funnel data, provide exactly 4 detailed insights:
   a) BIGGEST LEAK: Which stage-to-stage transition loses the most users? Cite exact user counts
      and percentages. Compare to industry benchmarks.
   b) ROOT CAUSE HYPOTHESIS: What product/UX issue most likely causes the biggest leak?
      Be specific (e.g., "search results page likely has poor filtering" not "improve UX").
   c) SECONDARY DROP-OFFS: Identify the 2nd and 3rd worst transitions. Are they related?
   d) SPECIFIC FIX: One concrete, implementable product change with expected conversion lift.

OUTPUT RULES:
- Use ONLY HTML tags for formatting: <h4>, <p>, <ul>, <li>, <strong>.
- Do NOT use markdown (no **, no ##, no - lists). Do NOT use emoji.
- Cite exact numbers from tool results. No vague statements like "significant drop-off".
- Keep insights concise but specific. Each insight should be 2-4 sentences max."""



def _calculate_funnel_overview(df, label):
    if df.empty:
        return {f"{label}_funnel_conversion": 0, f"{label}_search_dropoff": 0}

    ue = df.groupby("user_uuid")["event_name"].apply(set)
    total = len(ue)
    if total == 0:
        return {f"{label}_funnel_conversion": 0, f"{label}_search_dropoff": 0}
        
    # Funnel milestones
    step_start = sum(1 for e in ue if "app_start" in e or "Session Started" in e) # Proxy for start
    if step_start == 0: step_start = total # Fallback if app_start missing
    
    step_booked = sum(1 for e in ue if "payment_success" in e)
    
    # Search dropoff proxy (users who searched but didn't view results)
    step_search = sum(1 for e in ue if any(x in e for x in ["bus_search", "_bus-search_user-bus-search"]))
    step_results = sum(1 for e in ue if any(x in e for x in ["bus_result", "_bus-search_list"]))
    
    funnel_conv = round(step_booked / total * 100, 2)
    
    search_drop = 0
    if step_search > 0:
        search_drop = round((step_search - step_results) / step_search * 100, 1)

    return {
        f"{label}_funnel_conversion": funnel_conv,
        f"{label}_search_dropoff": search_drop
    }

def funnel_analysis_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        
        # --- Trend Analysis (T1 vs T2) ---
        min_t, max_t = df["event_time"].min(), df["event_time"].max()
        mid_t = min_t + (max_t - min_t) / 2
        
        df_t1 = df[df["event_time"] < mid_t]
        df_t2 = df[df["event_time"] >= mid_t]
        
        stats_t1 = _calculate_funnel_overview(df_t1, "T1")
        stats_t2 = _calculate_funnel_overview(df_t2, "T2")
        
        trend_context = f"""
        TREND ANALYSIS (First Half {min_t.strftime('%d-%b')} to {mid_t.strftime('%d-%b')} vs Second Half {mid_t.strftime('%d-%b')} to {max_t.strftime('%d-%b')}):
        - Overall Look-to-Book Conversion: {stats_t1['T1_funnel_conversion']}% -> {stats_t2['T2_funnel_conversion']}%
        - Search Stage Drop-off: {stats_t1['T1_search_dropoff']}% -> {stats_t2['T2_search_dropoff']}%
        
        CRITICAL: In your insights, you MUST explicitly compare T1 vs T2 to identify if the funnel is getting leakier or more efficient.
        """

        ctx = {}
        tools = create_query_tools(df, ctx) + create_funnel_tools(df, ctx)
        
        # Inject trend context
        final_prompt = SYSTEM_PROMPT + "\n\n" + trend_context
        
        insights, iters = run_agent(final_prompt, tools)
        chart_html = build_funnel_chart(ctx)

        result = {"insights": insights, "fig": chart_html,
                  "data": {**dict(enumerate(ctx.get("funnel_stages", []))), **stats_t1, **stats_t2}, # Flatten list for easy merge if needed, mostly for context
                  "funnel_stages": ctx.get("funnel_stages", []), # Keep original list structure for chart
                  "title": TITLE, "iterations": iters}

        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["funnel_stages"],
                       "insights": insights, "iterations": iters}, f, indent=2)

        print(f"  [OK] {METRIC_NAME} ({iters} tool-call iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
