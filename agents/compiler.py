import json
import os
import re
from agents.state import AnalyticsState
from agents.llm_client import call_llm


def compiler_node(state: AnalyticsState) -> dict:
    results = state.get("metric_results", {})
    errors = state.get("errors", [])
    summary = state.get("dataset_summary", {})

    print(f"[Compiler] Received {len(results)} metric results, {len(errors)} errors")

    prompt = f"""You are a senior product analytics consultant with 15 years of experience in mobile app growth,
retention, and monetisation. You specialise in bus/travel booking platforms.

You have received results from {len(results)} independent analysis agents that each examined a different
dimension of user behaviour within a bus ticket booking mobile app. Your job is to synthesize these
individual findings into a single, unified executive report that a VP of Product or CTO can act on.

Dataset summary (application events only, system events excluded):
{json.dumps(summary, indent=2)}

Individual metric results:
"""
    for name, data in results.items():
        insights = data.get("insights", "No insights")
        prompt += f"\n### {name}\n{insights}\n"

    prompt += """

CRITICAL FORMATTING RULES — violating any of these will cause rendering failures:
- Use ONLY HTML tags for formatting: <h3>, <h4>, <ul>, <li>, <p>, <strong>, <em>.
- Do NOT use markdown syntax anywhere. No **, no *, no ##, no - lists.
- Do NOT use emoji characters anywhere. Be professional and data-driven.
- Every paragraph must be wrapped in <p> tags.
- Every list must use <ul> and <li> tags.
- Section headers must use <h3> tags. Sub-headers use <h4>.
- Bold text must use <strong> tags, not **text**.
- Keep all numbers and percentages precise — round to 1 decimal place.

Produce a structured executive report with these exact sections:
1. <h3>Executive Summary</h3> — 3-4 sentences on overall product health, key KPIs, and trajectory.
2. <h3>Top 5 Critical Findings</h3> — Ranked by business impact. Each finding must cite specific numbers
   from the metric results. Format as numbered <li> items.
3. <h3>Cross-Metric Correlations</h3> — Patterns visible across 2+ metrics (e.g., high friction
   correlating with drop-off, retention tied to session depth). Cite which metrics you are correlating.
4. <h3>User Health Scorecard</h3> — Rate the app on: Acquisition, Activation, Retention, Revenue,
   Referral (AARRR framework). Give each a rating (Critical/Needs Work/Healthy) with 1 sentence why.
5. <h3>Strategic Recommendations</h3> — 5 prioritised actions. Each must have: the problem it solves,
   the expected impact (quantified), and implementation complexity (Low/Medium/High).
6. <h3>Quick Wins</h3> — 3 things implementable within a single sprint with highest ROI."""

    compiled_insights = call_llm(prompt)

    os.makedirs("outputs", exist_ok=True)
    html = _build_consolidated_html(compiled_insights, results, errors, summary)
    out_path = "outputs/analytics_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    compiled = {
        "executive_insights": compiled_insights,
        "metrics_completed": list(results.keys()),
        "metrics_failed": errors,
        "html_path": out_path,
    }

    print(f"[Compiler] Consolidated report -> {out_path}")
    return {"compiled_report": compiled}


def _clean_markdown(text: str) -> str:
    text = re.sub(r"```html?\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "\u23E9-\u23F3"
        "\u23F8-\u23FA"
        "\u200d\uFE0F"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    if not re.search(r"<[a-zA-Z]", text):
        lines = text.split("\n")
        html_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                html_lines.append(f"<h2>{line[2:]}</h2>")
            elif line.startswith("## "):
                html_lines.append(f"<h3>{line[3:]}</h3>")
            elif line.startswith("### "):
                html_lines.append(f"<h4>{line[4:]}</h4>")
            elif line.startswith("**") and line.endswith("**"):
                html_lines.append(f"<h4>{line[2:-2]}</h4>")
            elif line.startswith("- ") or line.startswith("* "):
                html_lines.append(f"<li>{line[2:]}</li>")
            else:
                html_lines.append(f"<p>{line}</p>")
        text = "\n".join(html_lines)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"^#{1,4}\s+(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    return text


def _build_consolidated_html(exec_insights, results, errors, summary):
    metric_sections = ""
    order = [
        "funnel_analysis", "dropoff_analysis", "friction_points", "session_metrics",
        "retention_analysis", "user_segmentation", "conversion_rates", "time_to_action",
        "event_frequency", "temporal_patterns", "user_journey_insights",
    ]
    for name in order:
        if name not in results:
            continue
        data = results[name]
        title = data.get("title", name.replace("_", " ").title())
        fig_html = data.get("fig", "")
        insights = data.get("insights", "")
        insights_html = _clean_markdown(insights)

        metric_sections += f"""
        <section class="metric-card" id="{name}">
            <div class="metric-header">
                <h2>{title}</h2>
            </div>
            <div class="chart-container">{fig_html}</div>
            <div class="insights-panel">
                <div class="insights-label">Analysis</div>
                <div class="insights-body">{insights_html}</div>
            </div>
        </section>
        """

    nav_links = ""
    for name in order:
        if name in results:
            label = results[name].get("title", name.replace("_", " ").title())
            nav_links += f'<a href="javascript:void(0)" class="nav-item" onclick="document.getElementById(\'{name}\').scrollIntoView({{behavior:\'smooth\', block:\'start\'}})">{label}</a>\n'

    exec_html = _clean_markdown(exec_insights)

    errors_html = ""
    if errors:
        errors_html = '<div class="error-banner"><h3>Pipeline Errors</h3><ul>' + "".join(f"<li>{e}</li>" for e in errors) + "</ul></div>"

    total_events = summary.get("total_events", 0)
    total_events_fmt = f"{total_events:,}" if isinstance(total_events, int) else str(total_events)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;450;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f4f4f5;
            --bg-tertiary: #e4e4e7;
            --bg-card: #ffffff;
            --border: #e4e4e7;
            --border-hover: #d4d4d8;
            --text-primary: #09090b;
            --text-secondary: #52525b;
            --text-muted: #71717a;
            --accent: #6366f1;
            --accent-dim: rgba(99,102,241,0.1);
            --accent-border: rgba(99,102,241,0.2);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.65;
            font-size: 14px;
            -webkit-font-smoothing: antialiased;
        }}

        /* Sidebar */
        .sidebar {{
            position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
            background: var(--bg-primary);
            border-right: 1px solid var(--border);
            padding: 32px 16px 24px;
            overflow-y: auto; z-index: 100;
            display: flex; flex-direction: column;
        }}
        .sidebar-brand {{
            font-size: 13px; font-weight: 600; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 32px;
            padding: 0 12px;
        }}
        .nav-section-label {{
            font-size: 11px; font-weight: 500; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1.2px;
            padding: 0 12px; margin: 24px 0 8px;
        }}
        .nav-item {{
            display: block; padding: 8px 12px; margin-bottom: 1px;
            color: var(--text-secondary); text-decoration: none;
            font-size: 13px; font-weight: 400;
            border-radius: 6px; transition: all 0.15s ease;
        }}
        .nav-item:hover {{
            background: var(--accent-dim); color: var(--text-primary);
        }}
        .nav-item.active {{
            background: var(--accent-dim); color: var(--accent);
            font-weight: 500;
        }}

        /* Main content */
        .main {{
            margin-left: 240px; padding: 48px 56px;
            max-width: 960px;
        }}

        /* Hero */
        .hero {{ margin-bottom: 48px; }}
        .hero h1 {{
            font-size: 28px; font-weight: 600; color: var(--text-primary);
            letter-spacing: -0.5px; margin-bottom: 6px;
        }}
        .hero .subtitle {{
            font-size: 14px; color: var(--text-muted); font-weight: 400;
        }}

        /* KPI strip */
        .kpi-row {{
            display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px;
            margin-bottom: 48px;
        }}
        .kpi {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px; padding: 20px;
        }}
        .kpi .value {{
            font-size: 24px; font-weight: 600; color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }}
        .kpi .label {{
            font-size: 11px; color: var(--text-muted); margin-top: 4px;
            text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500;
        }}

        /* Executive summary */
        .exec-summary {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px; padding: 32px;
            margin-bottom: 48px;
        }}
        .exec-summary > h2 {{
            font-size: 18px; font-weight: 600; color: var(--text-primary);
            margin-bottom: 20px; padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        .exec-summary h3 {{ font-size: 15px; color: var(--text-primary); margin: 20px 0 8px; font-weight: 600; }}
        .exec-summary h4 {{ font-size: 14px; color: var(--text-secondary); margin: 14px 0 6px; font-weight: 500; }}
        .exec-summary p {{ color: var(--text-secondary); margin-bottom: 8px; }}
        .exec-summary li {{ color: var(--text-secondary); margin-bottom: 5px; margin-left: 16px; }}
        .exec-summary strong {{ color: var(--text-primary); font-weight: 500; }}

        /* Metric card */
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin-bottom: 32px;
            overflow: hidden;
            transition: border-color 0.2s;
        }}
        .metric-card:hover {{ border-color: var(--border-hover); }}

        .metric-header {{
            padding: 24px 28px 0;
        }}
        .metric-header h2 {{
            font-size: 17px; font-weight: 600; color: var(--text-primary);
            letter-spacing: -0.2px;
        }}

        .chart-container {{
            padding: 8px 16px;
        }}
        .chart-container .plotly-graph-div {{ border-radius: 8px; }}

        .insights-panel {{
            border-top: 1px solid var(--border);
            padding: 24px 28px;
        }}
        .insights-label {{
            font-size: 11px; font-weight: 600; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
        }}
        .insights-body {{ color: var(--text-secondary); font-size: 13px; line-height: 1.7; }}
        .insights-body h3 {{ font-size: 14px; color: var(--text-primary); margin: 16px 0 6px; font-weight: 600; }}
        .insights-body h4 {{ font-size: 13px; color: var(--text-secondary); margin: 12px 0 4px; font-weight: 500; }}
        .insights-body p {{ margin-bottom: 6px; }}
        .insights-body li {{ margin-bottom: 4px; margin-left: 16px; }}
        .insights-body strong {{ color: var(--text-primary); font-weight: 500; }}

        /* Error */
        .error-banner {{
            background: rgba(239,68,68,0.06);
            border: 1px solid rgba(239,68,68,0.15);
            border-radius: 10px; padding: 20px; margin-bottom: 32px;
        }}
        .error-banner h3 {{ color: #fca5a5; font-size: 14px; margin-bottom: 8px; }}
        .error-banner li {{ color: #fb7185; margin-left: 16px; font-size: 13px; }}

        /* Footer */
        .footer {{
            text-align: center; padding: 32px 0; color: var(--text-muted);
            font-size: 12px; border-top: 1px solid var(--border); margin-top: 48px;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 5px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--bg-tertiary); border-radius: 3px; }}

        html {{ scroll-behavior: smooth; }}

        @media (max-width: 900px) {{
            .sidebar {{ display: none; }}
            .main {{ margin-left: 0; padding: 24px 20px; }}
            .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="sidebar-brand">Analytics</div>
        <a href="javascript:void(0)" class="nav-item active" onclick="document.getElementById('executive').scrollIntoView({{behavior:'smooth', block:'start'}})">Executive Summary</a>
        <div class="nav-section-label">Metrics</div>
        {nav_links}
    </nav>

    <main class="main">
        <div class="hero">
            <h1>Analytics Report</h1>
            <p class="subtitle">{summary.get('date_range_str', 'Date Range Unknown')} • {summary.get('total_users', '?')} users • {total_events_fmt} events</p>
        </div>

        <div class="kpi-row">
            <div class="kpi"><div class="value">{summary.get('total_users', '?')}</div><div class="label">Users</div></div>
            <div class="kpi"><div class="value">{total_events_fmt}</div><div class="label">Events</div></div>
            <div class="kpi"><div class="value">{summary.get('days_covered', '?')}</div><div class="label">Days Covered</div></div>
            <div class="kpi"><div class="value">{summary.get('peak_day', 'N/A')}</div><div class="label">Peak Day</div></div>
            <div class="kpi"><div class="value">{summary.get('total_event_types', '?')}</div><div class="label">Event Types</div></div>
            <div class="kpi"><div class="value">{len(results)}/{len(order)}</div><div class="label">Metrics Completed</div></div>
        </div>

        {errors_html}

        <section class="exec-summary" id="executive">
            <h2>Executive Summary</h2>
            {exec_html}
        </section>

        {metric_sections}

        <div class="footer">
            Generated by LangGraph Multi-Agent Pipeline
        </div>
    </main>

    <script>
        const navItems = document.querySelectorAll('.nav-item');
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    navItems.forEach(n => n.classList.remove('active'));
                    const id = entry.target.id;
                    navItems.forEach(link => {{
                        if (link.getAttribute('onclick') && link.getAttribute('onclick').includes(id)) {{
                            link.classList.add('active');
                        }}
                    }});
                }}
            }});
        }}, {{ rootMargin: '-20% 0px -70% 0px' }});
        document.querySelectorAll('section[id]').forEach(s => observer.observe(s));

        document.addEventListener('click', function(e) {{
            if (e.target.tagName === 'A') {{
                e.preventDefault();
            }}
        }});
    </script>
</body>
</html>"""
