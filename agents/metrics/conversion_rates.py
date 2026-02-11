import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_conversion_tools
from agents.charts import build_conversion_chart

METRIC_NAME = "conversion_rates"
TITLE = "Conversion Rate Analysis"
SYSTEM_PROMPT = """You are a conversion optimization expert specialising in payment flows
and monetisation for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Conversion is measured as: users who completed payment_success / total users.
- Payment success rate = payment_success / payment_initiate (measures payment flow reliability).
- Push conversion = users who saw push notifications AND later converted / total push users.
- For travel apps, overall conversion rates of 2-5% are typical. Payment success rates
  above 80% indicate a reliable payment flow. Below 70% suggests major payment issues.
- Payment failure events include: payment_failed, paymentFailed_backpressed,
  paymentFailed_pending_vbv.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_conversions to get overall conversion, payment success rate, push conversion.
3. Provide analysis covering:
   a) CONVERSION HEALTH: Is the overall conversion rate healthy? How does it compare
      to the 2-5% industry benchmark? Cite exact percentages.
   b) PAYMENT FAILURE ANALYSIS: What is the payment success rate? What could cause failures?
      Cite exact failure counts and rates.
   c) PUSH NOTIFICATION ROI: Are push notifications driving conversions? What is the
      push-to-conversion rate compared to organic users?
   d) CONVERSION TACTICS: 3 specific, implementable changes to improve conversion,
      each with expected impact quantified.

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact numbers. No vague language."""



def _calculate_conversion(df, label):
    if df.empty:
        return {f"{label}_conversion": 0, f"{label}_pay_success": 0}

    users = df["user_uuid"].nunique()
    if users == 0:
        return {f"{label}_conversion": 0, f"{label}_pay_success": 0}

    # Identify users who performed specific actions
    ue = df.groupby("user_uuid")["event_name"].apply(set)
    
    converters = sum(1 for e in ue if "payment_success" in e)
    attempters = sum(1 for e in ue if e & {"payment_initiate", "PaymentPage_payment initiated"})
    
    conv_rate = round(converters / users * 100, 2)
    pay_rate = round(converters / attempters * 100, 2) if attempters > 0 else 0
    
    return {
        f"{label}_conversion": conv_rate,
        f"{label}_pay_success": pay_rate
    }

def conversion_rates_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        
        # --- Trend Analysis (T1 vs T2) ---
        min_t, max_t = df["event_time"].min(), df["event_time"].max()
        mid_t = min_t + (max_t - min_t) / 2
        
        df_t1 = df[df["event_time"] < mid_t]
        df_t2 = df[df["event_time"] >= mid_t]
        
        stats_t1 = _calculate_conversion(df_t1, "T1")
        stats_t2 = _calculate_conversion(df_t2, "T2")
        
        trend_context = f"""
        TREND ANALYSIS (First Half {min_t.strftime('%d-%b')} to {mid_t.strftime('%d-%b')} vs Second Half {mid_t.strftime('%d-%b')} to {max_t.strftime('%d-%b')}):
        - Overall Conversion: {stats_t1['T1_conversion']}% -> {stats_t2['T2_conversion']}%
        - Payment Success Rate: {stats_t1['T1_pay_success']}% -> {stats_t2['T2_pay_success']}%
        
        CRITICAL: In your insights, you MUST explicitly compare T1 vs T2 to identify if conversion performance is improving or degrading.
        """

        ctx = {}
        tools = create_query_tools(df, ctx) + create_conversion_tools(df, ctx)
        
        # Inject trend context
        final_prompt = SYSTEM_PROMPT + "\n\n" + trend_context
        
        insights, iters = run_agent(final_prompt, tools)
        chart_html = build_conversion_chart(ctx)
        
        # Merge trend data into result
        result = {"insights": insights, "fig": chart_html,
                  "data": {**ctx.get("conversion_data", {}), **stats_t1, **stats_t2}, 
                  "title": TITLE, "iterations": iters}
                  
        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)
        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
