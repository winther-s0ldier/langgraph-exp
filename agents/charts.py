import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np


def build_funnel_chart(ctx):
    stages = ctx.get("funnel_stages", [])
    if not stages:
        return ""
    palette = ["#818cf8","#7c3aed","#6d28d9","#a78bfa","#c084fc","#22d3ee","#2dd4bf","#34d399"]
    fig = go.Figure(go.Funnel(
        y=[s["stage"] for s in stages], x=[s["users"] for s in stages],
        textinfo="value+percent initial+percent previous",
        marker=dict(color=palette[:len(stages)]),
        connector=dict(line=dict(color="#4c1d95", width=1)),
    ))
    fig.update_layout(title=dict(text="Booking Funnel", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=520,
                      margin=dict(l=200, r=40, t=50, b=40))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_dropoff_chart(ctx):
    drops = ctx.get("dropoffs", [])
    counts = ctx.get("dropoff_counts", [])
    labels = ctx.get("dropoff_labels", [])
    if not drops:
        return ""
    fig = go.Figure(go.Waterfall(
        orientation="v", x=[labels[0]] + [d["to"] for d in drops],
        y=[counts[0]] + [-d["lost"] for d in drops],
        connector=dict(line=dict(color="#475569", width=1)),
        increasing=dict(marker=dict(color="#818cf8")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#22d3ee")),
        text=[str(counts[0])] + [f"-{d['lost']} ({d['pct']}%)" for d in drops],
        textposition="outside", textfont=dict(size=11, color="#e2e8f0"),
    ))
    fig.update_layout(title=dict(text="User Drop-off Waterfall", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=480,
                      yaxis_title="Users", xaxis=dict(tickangle=-30))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_friction_chart(ctx):
    friction = ctx.get("friction_events", [])[:10]
    if not friction:
        return ""
    fig = go.Figure(go.Bar(
        y=[f["event"] for f in friction], x=[f["score"] for f in friction], orientation="h",
        marker=dict(color=[f["score"] for f in friction],
                    colorscale=[[0,"#fbbf24"],[0.5,"#f97316"],[1,"#ef4444"]],
                    colorbar=dict(title=dict(text="Score", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"))),
        text=[f"{f['repeat_rate']}% rep | {f['avg_per_session']}x" for f in friction],
        textposition="auto", textfont=dict(size=10, color="#f8fafc"),
    ))
    fig.update_layout(title=dict(text="Friction Points", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=480,
                      yaxis=dict(autorange="reversed"), margin=dict(l=200, t=50))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_session_chart(ctx):
    stats = ctx.get("session_stats", {})
    durs = ctx.get("session_durations", [])
    evts = ctx.get("session_event_counts", [])
    depths = ctx.get("session_depths", [])
    spu = ctx.get("spu_dist", [])
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Duration (s)", "Events/Session", "Depth", "Sessions/User"),
                        vertical_spacing=0.12, horizontal_spacing=0.1)
    fig.add_trace(go.Histogram(x=durs, nbinsx=40, marker_color="#818cf8", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Violin(y=evts, box_visible=True, meanline_visible=True, marker_color="#7c3aed"), row=1, col=2)
    fig.add_trace(go.Histogram(x=depths, nbinsx=25, marker_color="#22d3ee", opacity=0.85), row=2, col=1)
    fig.add_trace(go.Histogram(x=spu, nbinsx=20, marker_color="#f59e0b", opacity=0.85), row=2, col=2)
    n = stats.get("total_sessions", "?")
    bp = stats.get("bounce_pct", "?")
    fig.update_layout(title=dict(text=f"Sessions: {n:,} | Bounce: {bp}%", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=11, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=560, showlegend=False)
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_retention_chart(ctx):
    r = ctx.get("retention", {})
    matrix, labels = r.get("matrix",[]), r.get("labels",[])
    if not matrix:
        return ""
    mw = r.get("max_weeks", 5)
    text_m = [[f"{v}%" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=[f"Week {w}" for w in range(mw)], y=labels,
        colorscale=[[0,"#1e1b4b"],[0.3,"#4338ca"],[0.6,"#818cf8"],[1,"#c4b5fd"]],
        text=text_m, texttemplate="%{text}", textfont=dict(size=12, color="#f8fafc"),
        colorbar=dict(title=dict(text="Retention %", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8")),
    ))
    fig.update_layout(title=dict(text=f"Cohort Retention | W1: {r.get('w1_pct',0)}%", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                      height=max(380, 50*len(labels)+100), margin=dict(l=130, t=50))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_segmentation_chart(ctx):
    profiles = ctx.get("segments", [])
    scatter = ctx.get("scatter", {})
    if not profiles:
        return ""
    palette = ["#818cf8","#22d3ee","#34d399","#fbbf24","#f87171","#e879f9","#6b7280"]
    fig = go.Figure()
    xs, ys, ls = scatter.get("x",[]), scatter.get("y",[]), scatter.get("labels",[])
    for p in profiles:
        sid = p["id"]
        idx = [i for i, l in enumerate(ls) if l == sid]
        c = palette[6] if sid == -1 else palette[sid % 6]
        fig.add_trace(go.Scatter(x=[xs[i] for i in idx], y=[ys[i] for i in idx],
                                 mode="markers", name=f"{p['label']} ({p['size']})",
                                 marker=dict(color=c, size=6, opacity=0.7)))
    fig.update_layout(title=dict(text=f"User Segments (DBSCAN) | {len(profiles)} clusters", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=500,
                      xaxis_title="PCA 1", yaxis_title="PCA 2",
                      legend=dict(bgcolor="rgba(15,23,42,0.8)", font=dict(size=11)),
                      xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_conversion_chart(ctx):
    d = ctx.get("conversion_data", {})
    if not d:
        return ""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"bar"},{"type":"pie"}]],
                        subplot_titles=("Conversion Funnel","Payment Outcome"), column_widths=[0.55,0.45])
    fig.add_trace(go.Bar(x=["Total","Attempted","Success"], y=[d["total"],d["attempted"],d["converters"]],
                         marker=dict(color=["#818cf8","#f59e0b","#34d399"]),
                         text=[d["total"],d["attempted"],d["converters"]], textposition="auto",
                         textfont=dict(color="#f8fafc", size=13)), row=1, col=1)
    fig.add_trace(go.Pie(labels=["Success","Failed","No Payment"],
                         values=[d["converters"],d["failed"],max(0,d["total"]-d["attempted"])],
                         marker=dict(colors=["#34d399","#ef4444","#475569"], line=dict(color="#0f172a",width=2)),
                         hole=0.5, textinfo="percent+label", textfont=dict(size=11, color="#f8fafc")), row=1, col=2)
    fig.update_layout(title=dict(text=f"Conversion: {d['conv_pct']}%", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=440, showlegend=False)
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_latency_chart(ctx):
    pairs = ctx.get("latency_pairs", [])
    if not pairs:
        return ""
    colors = ["#818cf8","#22d3ee","#34d399","#fbbf24","#f87171"]
    fills = ["rgba(129,140,248,0.3)","rgba(34,211,238,0.3)","rgba(52,211,153,0.3)",
             "rgba(251,191,36,0.3)","rgba(248,113,113,0.3)"]
    fig = go.Figure()
    for i, p in enumerate(pairs):
        label = f"{p['from']} -> {p['to']}"
        fig.add_trace(go.Violin(y=p.get("values",[]), name=label, box_visible=True, meanline_visible=True,
                                marker_color=colors[i%5], line_color=colors[i%5], fillcolor=fills[i%5]))
    fig.update_layout(title=dict(text="Time-to-Action Latency (s)", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=480,
                      yaxis_title="Seconds", showlegend=False, yaxis=dict(gridcolor="#334155"))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_frequency_chart(ctx):
    fd = ctx.get("frequency_data", {})
    top = fd.get("top_events", [])
    if not top:
        return ""
    import pandas as pd
    tmap = pd.DataFrame(top)
    tmap["parent"] = "All Events"
    fig = px.treemap(tmap, path=["parent","event"], values="count",
                     color="pct", color_continuous_scale=[[0,"#1e1b4b"],[0.3,"#4338ca"],[0.7,"#818cf8"],[1,"#c4b5fd"]],
                     title="Event Frequency (Top 20)")
    fig.update_layout(font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=520,
                      margin=dict(t=50, l=10, r=10, b=10))
    fig.update_traces(textfont=dict(color="#f8fafc"), marker_line_width=1, marker_line_color="#0f172a")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_temporal_chart(ctx):
    d = ctx.get("temporal", {})
    matrix = ctx.get("temporal_matrix", [])
    if not matrix:
        return ""
    DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=[f"{h:02d}:00" for h in range(24)], y=DAYS,
        colorscale=[[0,"#0f172a"],[0.2,"#312e81"],[0.4,"#4338ca"],[0.6,"#6366f1"],[0.8,"#a78bfa"],[1,"#e9d5ff"]],
        colorbar=dict(title=dict(text="Events", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8")),
    ))
    pk_d, pk_h, ratio = d.get("peak_day","?"), d.get("peak_hour","?"), d.get("ratio","?")
    fig.update_layout(title=dict(text=f"Usage Heatmap | Peak: {pk_d} {pk_h}:00 | Ratio: {ratio}x", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=12, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=400, margin=dict(l=100, t=50))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_user_journey_chart(ctx):
    metrics = ctx.get("user_journey_stats", {})
    if not metrics:
        return ""
    
    types = metrics.get("user_types", {})
    funnel = metrics.get("funnel_counts", {})
    issues = metrics.get("friction_counts", {})
    
    fig = make_subplots(rows=1, cols=3, specs=[[{"type":"domain"},{"type":"bar"},{"type":"pie"}]],
                        subplot_titles=("User Segments", "Journey Funnel", "Friction Impact"),
                        column_widths=[0.33, 0.33, 0.33])
    
    # User Segments Donut
    labels = ["Browsers", "Shoppers", "Attempters", "Bookers"]
    values = [types.get(k.lower(), 0) for k in labels]
    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4,
                         marker=dict(colors=["#94a3b8", "#60a5fa", "#fbbf24", "#34d399"]),
                         textinfo="percent+label"), row=1, col=1)
    
    # Funnel Bar Chart
    stages = ["searched", "selected", "pay_init", "booked"]
    counts = [funnel.get(s, 0) for s in stages]
    fig.add_trace(go.Bar(x=stages, y=counts, marker_color="#818cf8",
                         text=counts, textposition="auto"), row=1, col=2)
    
    # Issues Pie Chart
    issue_labels = ["Payment Failure", "App Error", "Clean Journey"]
    fail = issues.get("faced_payment_failure", 0)
    err = issues.get("faced_app_error", 0)
    clean = max(0, metrics.get("total_analyzed_users", 0) - fail - err)
    
    fig.add_trace(go.Pie(labels=issue_labels, values=[fail, err, clean],
                         marker=dict(colors=["#ef4444", "#f43f5e", "#22c55e"]),
                         textinfo="percent+label"), row=1, col=3)
    
    fig.update_layout(title=dict(text="User Journey Overview", font=dict(size=18, color="#e2e8f0")),
                      font=dict(family="Inter,sans-serif", size=11, color="#cbd5e1"),
                      paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=400, showlegend=False)
    return fig.to_html(include_plotlyjs=False, full_html=False)
