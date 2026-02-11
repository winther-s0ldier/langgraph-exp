import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_session_tools
from agents.charts import build_session_chart

METRIC_NAME = "session_metrics"
TITLE = "Session Behaviour Metrics"
SYSTEM_PROMPT = """You are a product analyst specialising in session behaviour and user
engagement for mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- A session is a continuous period of user activity, detected by session marker events
  (Session Started, Journey Started, App Installed, User Login).
- Key session health metrics: total sessions, average duration, bounce rate (sessions with
  <= 2 events), session depth (unique event types per session), sessions per user.
- For a booking app, healthy sessions typically last 3-10 minutes. Bounce rates below 30%
  are considered good. Average depth of 5+ unique event types indicates engaged users.
- Sessions per user > 2 suggests repeat usage and healthy retention.

TASK:
1. Call get_dataset_summary for dataset context.
2. Call compute_session_stats to get session-level metrics.
3. Provide exactly 4 insights:
   a) SESSION HEALTH: Overall assessment of session quality. Compare avg duration, depth,
      and bounce rate against the benchmarks above. Are sessions healthy or concerning?
   b) ENGAGEMENT DEPTH: What does the event depth distribution tell us? Are most users
      exploring the app or leaving after 1-2 actions?
   c) BOUNCE ANALYSIS: What is the bounce rate and what likely causes it? Cite the exact
      percentage and hypothesise causes (e.g., slow load, poor onboarding).
   d) SESSION TARGETS: Recommend specific session duration and depth targets the product
      team should aim for, based on the current data and industry standards.

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Cite exact numbers from tool results. No vague language."""


def _calculate_stats(df, label):
    if df.empty:
        return {f"{label}_sessions": 0, f"{label}_avg_dur": 0, f"{label}_bounce": 0}
    
    s = df.sort_values(["user_uuid", "event_time"]).copy()
    s["is_ss"] = s["event_name"].isin(["Session Started", "Journey Started", "App Installed", "User Login"])
    s["sid"] = s.groupby("user_uuid")["is_ss"].cumsum()
    s["skey"] = s["user_uuid"] + "_" + s["sid"].astype(str)
    
    sess = s.groupby("skey").agg(
        events=("event_name", "count"),
        start=("event_time", "min"),
        end=("event_time", "max")
    )
    sess["dur"] = (sess["end"] - sess["start"]).dt.total_seconds()
    sess = sess[sess["events"] > 1]  # Filter single-event noise
    
    n = len(sess)
    if n == 0:
        return {f"{label}_sessions": 0, f"{label}_avg_dur": 0, f"{label}_bounce": 0}
        
    return {
        f"{label}_sessions": n,
        f"{label}_avg_dur": round(sess["dur"].mean(), 1),
        f"{label}_bounce": round((sess["events"] <= 2).sum() / n * 100, 1)
    }

def session_metrics_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        
        # --- Trend Analysis (T1 vs T2) ---
        min_t, max_t = df["event_time"].min(), df["event_time"].max()
        mid_t = min_t + (max_t - min_t) / 2
        
        df_t1 = df[df["event_time"] < mid_t]
        df_t2 = df[df["event_time"] >= mid_t]
        
        stats_t1 = _calculate_stats(df_t1, "T1")
        stats_t2 = _calculate_stats(df_t2, "T2")
        
        trend_context = f"""
        TREND ANALYSIS (First Half {min_t.strftime('%d-%b')} to {mid_t.strftime('%d-%b')} vs Second Half {mid_t.strftime('%d-%b')} to {max_t.strftime('%d-%b')}):
        - Sessions: {stats_t1['T1_sessions']} -> {stats_t2['T2_sessions']}
        - Avg Duration: {stats_t1['T1_avg_dur']}s -> {stats_t2['T2_avg_dur']}s
        - Bounce Rate: {stats_t1['T1_bounce']}% -> {stats_t2['T2_bounce']}%
        
        CRITICAL: In your insights, you MUST explicitly compare T1 vs T2 to identify if user behavior is improving or degrading.
        """
        
        ctx = {}
        tools = create_query_tools(df, ctx) + create_session_tools(df, ctx)
        
        # Inject trend context into the prompt
        final_prompt = SYSTEM_PROMPT + "\n\n" + trend_context
        
        insights, iters = run_agent(final_prompt, tools)
        chart_html = build_session_chart(ctx)

        result = {"insights": insights, "fig": chart_html,
                  "data": {**ctx.get("session_stats", {}), **stats_t1, **stats_t2}, 
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
