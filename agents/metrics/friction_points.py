import json, os, pandas as pd
from agents.state import AnalyticsState
from agents.agent_runner import run_agent
from agents.tools import create_query_tools, create_friction_tools
from agents.charts import build_friction_chart

METRIC_NAME = "friction_points"
TITLE = "Friction Point Analysis"
SYSTEM_PROMPT = """You are a UX analytics expert specialising in detecting friction and
usability issues in mobile apps. You are analysing a bus ticket booking app.
The dataset contains ONLY application-level events (system events excluded).

CONTEXT:
- Friction is detected when users repeat the same event multiple times within a single session,
  indicating confusion, errors, or poor UX (e.g., tapping a button repeatedly because it did
  not respond, re-searching because results were unsatisfactory).
- The friction score is calculated as: repeat_rate * log(1 + avg_repeats_per_session).
  Higher scores indicate more severe friction.
- Events with high repeat rates but low total volume may be noise. Focus on events with
  both high repeat rates AND meaningful volume (>= 20 occurrences).

TASK:
1. Call get_dataset_summary for dataset context.
2. Call detect_repeated_events with min_total=20 to find friction events.
3. For the top 3 friction points (by score), provide:
   a) WHAT IT MEANS: What UX issue does this repeated event indicate? Be specific about the
      user scenario (e.g., "Users are repeatedly tapping seat selection, likely because the
      seat map is unresponsive or confusing").
   b) SEVERITY: How many users and sessions are affected? Cite exact numbers.
   c) PROPOSED FIX: One specific, implementable solution.
   d) EXPECTED IMPACT: What metric improvement you expect (e.g., "reduce bounce rate by X%").

OUTPUT RULES:
- Use ONLY HTML tags: <h4>, <p>, <ul>, <li>, <strong>. No markdown, no emoji.
- Structure each friction point as a separate subsection with <h4> tags.
- Cite exact numbers from tool results."""


def friction_points_node(state: AnalyticsState) -> dict:
    try:
        df = pd.read_csv(state["dataset_path"])
        df["event_time"] = pd.to_datetime(df["event_time"], format="mixed", utc=True)
        df = df[df["category"] == "application"]
        ctx = {}
        tools = create_query_tools(df, ctx) + create_friction_tools(df, ctx)
        insights, iters = run_agent(SYSTEM_PROMPT, tools)
        chart_html = build_friction_chart(ctx)

        result = {"insights": insights, "fig": chart_html,
                  "data": ctx.get("friction_events", []), "title": TITLE, "iterations": iters}

        os.makedirs("outputs/json", exist_ok=True)
        with open(f"outputs/json/{METRIC_NAME}.json", "w") as f:
            json.dump({"metric": METRIC_NAME, "title": TITLE, "data": result["data"],
                       "insights": insights, "iterations": iters}, f, indent=2)

        print(f"  [OK] {METRIC_NAME} ({iters} iterations)")
        return {"metric_results": {METRIC_NAME: result}}
    except Exception as e:
        print(f"  [FAIL] {METRIC_NAME}: {str(e)[:200]}")
        return {"errors": [f"{METRIC_NAME}: {str(e)[:200]}"]}
