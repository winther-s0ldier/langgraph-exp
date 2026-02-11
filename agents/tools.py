import json
import pandas as pd
import numpy as np
from langchain_core.tools import tool

SESSION_MARKERS = {"Session Started", "Journey Started", "App Installed", "User Login"}


def create_query_tools(df, ctx):
    @tool
    def get_dataset_summary() -> str:
        """Get dataset summary: total events, users, event types, date range, categories."""
        info = {
            "total_events": len(df),
            "total_users": int(df["user_uuid"].nunique()),
            "event_types": int(df["event_name"].nunique()),
            "date_start": str(df["event_time"].min()),
            "date_end": str(df["event_time"].max()),
            "categories": {k: int(v) for k, v in df["category"].value_counts().items()},
        }
        ctx["dataset_summary"] = info
        return json.dumps(info, indent=2)

    @tool
    def list_event_names(category: str = "") -> str:
        """List unique event names, optionally filtered by category."""
        sub = df[df["category"] == category] if category else df
        return json.dumps(sorted(sub["event_name"].unique().tolist()))

    @tool
    def count_users_with_events(event_names: list[str]) -> str:
        """Count users who triggered at least one of the given events."""
        ue = df.groupby("user_uuid")["event_name"].apply(set)
        t = set(event_names)
        c = sum(1 for e in ue if e & t)
        return json.dumps({"matching": c, "total": len(ue), "pct": round(c / len(ue) * 100, 2)})

    @tool
    def get_top_events(n: int = 20) -> str:
        """Get top N events by frequency."""
        ec = df["event_name"].value_counts().head(n)
        return json.dumps([{"event": e, "count": int(c), "pct": round(c / len(df) * 100, 2)} for e, c in ec.items()])

    return [get_dataset_summary, list_event_names, count_users_with_events, get_top_events]


def create_funnel_tools(df, ctx):
    @tool
    def compute_funnel(stages: list[dict]) -> str:
        """Compute multi-stage funnel. stages: list of {name, events} dicts. Returns per-stage user counts and conversion rates."""
        ue = df.groupby("user_uuid")["event_name"].apply(set)
        total = len(ue)
        result = []
        for s in stages:
            c = sum(1 for e in ue if e & set(s["events"]))
            result.append({"stage": s["name"], "users": c, "pct": round(c / total * 100, 1)})
        for i in range(1, len(result)):
            p = result[i - 1]["users"]
            result[i]["conv_from_prev"] = round(result[i]["users"] / p * 100, 1) if p else 0
            result[i]["lost"] = p - result[i]["users"]
        ctx["funnel_stages"] = result
        return json.dumps(result, indent=2)
    return [compute_funnel]


def create_dropoff_tools(df, ctx):
    @tool
    def compute_dropoffs(stages: list[dict]) -> str:
        """Compute drop-off between funnel stages. stages: list of {name, events}."""
        ue = df.groupby("user_uuid")["event_name"].apply(set)
        counts = [sum(1 for e in ue if e & set(s["events"])) for s in stages]
        labels = [s["name"] for s in stages]
        drops = []
        for i in range(1, len(counts)):
            lost = counts[i - 1] - counts[i]
            drops.append({"from": labels[i-1], "to": labels[i], "lost": lost,
                          "pct": round(lost / counts[i-1] * 100, 1) if counts[i-1] else 0})
        ctx["dropoffs"] = drops
        ctx["dropoff_counts"] = counts
        ctx["dropoff_labels"] = labels
        return json.dumps(drops, indent=2)
    return [compute_dropoffs]


def create_friction_tools(df, ctx):
    @tool
    def detect_repeated_events(min_total: int = 20) -> str:
        """Detect events repeated within sessions (friction). Returns ranked by composite friction score."""
        s = df.sort_values(["user_uuid", "event_time"]).copy()
        s["is_ss"] = s["event_name"].isin(SESSION_MARKERS)
        s["sid"] = s.groupby("user_uuid")["is_ss"].cumsum()
        s["skey"] = s["user_uuid"] + "_" + s["sid"].astype(str)
        app = s[s["category"] == "application"].copy()
        app["prev"] = app.groupby("skey")["event_name"].shift(1)
        app["is_rep"] = app["event_name"] == app["prev"]
        agg = app.groupby("event_name")["is_rep"].agg(["sum", "count"])
        agg.columns = ["repeats", "total"]
        agg["rate"] = (agg["repeats"] / agg["total"] * 100).round(1)
        agg = agg[agg["total"] >= min_total].sort_values("rate", ascending=False).head(12)
        friction = []
        for evt, row in agg.iterrows():
            avg = round(float(app[app["event_name"] == evt].groupby("skey").size().mean()), 1)
            score = round(row["rate"] * float(np.log1p(avg)), 1)
            friction.append({"event": evt, "repeat_rate": float(row["rate"]),
                             "avg_per_session": avg, "total": int(row["total"]), "score": score})
        friction.sort(key=lambda x: x["score"], reverse=True)
        ctx["friction_events"] = friction
        return json.dumps(friction, indent=2)
    return [detect_repeated_events]


def create_session_tools(df, ctx):
    @tool
    def compute_session_stats() -> str:
        """Compute session stats: total, avg duration, bounce rate, depth, sessions per user."""
        s = df.sort_values(["user_uuid", "event_time"]).copy()
        s["is_ss"] = s["event_name"].isin(SESSION_MARKERS)
        s["sid"] = s.groupby("user_uuid")["is_ss"].cumsum()
        s["skey"] = s["user_uuid"] + "_" + s["sid"].astype(str)
        sess = s.groupby("skey").agg(events=("event_name","count"), unique=("event_name","nunique"),
                                     start=("event_time","min"), end=("event_time","max"), user=("user_uuid","first"))
        sess["dur"] = (sess["end"] - sess["start"]).dt.total_seconds()
        sess = sess[sess["events"] > 1]
        n = len(sess)
        stats = {"total_sessions": n, "avg_dur_sec": round(float(sess["dur"].mean()),1),
                 "median_dur_sec": round(float(sess["dur"].median()),1),
                 "avg_events": round(float(sess["events"].mean()),1),
                 "avg_depth": round(float(sess["unique"].mean()),1),
                 "bounce_pct": round(float((sess["events"]<=2).sum())/n*100,1) if n else 0,
                 "sessions_per_user": round(n/sess["user"].nunique(),1)}
        ctx["session_stats"] = stats
        ctx["session_durations"] = sess["dur"].clip(upper=sess["dur"].quantile(0.95)).tolist()
        ctx["session_event_counts"] = sess["events"].clip(upper=sess["events"].quantile(0.95)).tolist()
        ctx["session_depths"] = sess["unique"].tolist()
        ctx["spu_dist"] = sess.groupby("user").size().clip(upper=20).tolist()
        return json.dumps(stats, indent=2)
    return [compute_session_stats]


def create_retention_tools(df, ctx):
    @tool
    def compute_retention(max_weeks: int = 5) -> str:
        """Compute weekly cohort retention matrix."""
        uf = df.groupby("user_uuid")["event_time"].min().reset_index()
        uf.columns = ["user_uuid", "first"]
        uf["cw"] = uf["first"].dt.isocalendar().week.astype(int)
        m = df.merge(uf, on="user_uuid")
        m["ws"] = ((m["event_time"] - m["first"]).dt.total_seconds() / 604800).astype(int).clip(lower=0)
        mw = min(int(m["ws"].max()) + 1, max_weeks)
        matrix, labels = [], []
        for cw in sorted(m["cw"].unique()):
            users = set(uf[uf["cw"] == cw]["user_uuid"])
            if len(users) < 5: continue
            row = [round(m[(m["cw"]==cw)&(m["ws"]==w)]["user_uuid"].nunique()/len(users)*100,1) for w in range(mw)]
            matrix.append(row)
            labels.append(f"W{cw} (n={len(users)})")
        total = df["user_uuid"].nunique()
        ret1 = m[m["ws"]>=1]["user_uuid"].nunique()
        ctx["retention"] = {"matrix": matrix, "labels": labels, "max_weeks": mw,
                            "w1_pct": round(ret1/total*100,1) if total else 0}
        return json.dumps(ctx["retention"], indent=2)
    return [compute_retention]


def create_segmentation_tools(df, ctx):
    @tool
    def cluster_users(eps: float = 1.2, min_samples: int = 10) -> str:
        """Cluster users with DBSCAN on behavioral features. Returns segment profiles."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        s = df.sort_values(["user_uuid","event_time"]).copy()
        s["is_ss"] = s["event_name"].isin(SESSION_MARKERS)
        s["sid"] = s.groupby("user_uuid")["is_ss"].cumsum()
        bk = {"payment_success","book_ticket","Booking_to_Ticket","payment_initiate"}
        feats = s.groupby("user_uuid").agg(total=("event_name","count"),unique=("event_name","nunique"),
                                           sessions=("sid","max"),span=("event_time",lambda x:(x.max()-x.min()).days)).reset_index()
        feats["booked"] = feats["user_uuid"].isin(s[s["event_name"].isin(bk)]["user_uuid"].unique()).astype(int)
        feats["eps_r"] = (feats["total"]/feats["sessions"].clip(lower=1)).round(1)
        feats["div"] = (feats["unique"]/feats["total"]).round(3)
        X = StandardScaler().fit_transform(feats[["total","unique","sessions","span","eps_r","div"]].fillna(0))
        feats["seg"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        coords = PCA(n_components=2).fit_transform(X)
        feats["px"], feats["py"] = coords[:,0], coords[:,1]
        profiles = []
        for seg in sorted(feats["seg"].unique()):
            sd = feats[feats["seg"]==seg]
            profiles.append({"label": "Outliers" if seg==-1 else f"Segment {seg}", "id":int(seg),
                             "size":len(sd), "pct":round(len(sd)/len(feats)*100,1),
                             "avg_events":round(float(sd["total"].mean()),1),
                             "booking_rate":round(float(sd["booked"].mean())*100,1)})
        ctx["segments"] = profiles
        ctx["scatter"] = {"x":feats["px"].tolist(),"y":feats["py"].tolist(),"labels":feats["seg"].tolist()}
        return json.dumps(profiles, indent=2)
    return [cluster_users]


def create_conversion_tools(df, ctx):
    @tool
    def compute_conversions() -> str:
        """Compute conversion metrics: overall, payment success rate, push conversion."""
        ue = df.groupby("user_uuid")["event_name"].apply(set)
        t = len(ue)
        conv = sum(1 for e in ue if e & {"payment_success"})
        att = sum(1 for e in ue if e & {"payment_initiate","PaymentPage_payment initiated"})
        fail = sum(1 for e in ue if e & {"payment_failed","paymentFailed_backpressed","paymentFailed_pending_vbv"})
        pu = sum(1 for e in ue if e & {"Push Click","Push Impression"})
        pc = sum(1 for e in ue if (e & {"Push Click","Push Impression"}) and (e & {"payment_success"}))
        data = {"total":t,"converters":conv,"conv_pct":round(conv/t*100,2) if t else 0,
                "attempted":att,"failed":fail,"pay_success_pct":round(conv/att*100,2) if att else 0,
                "push_users":pu,"push_conv":pc,"push_pct":round(pc/pu*100,2) if pu else 0}
        ctx["conversion_data"] = data
        return json.dumps(data, indent=2)
    return [compute_conversions]


def create_latency_tools(df, ctx):
    @tool
    def compute_latency(from_event: str, to_event: str) -> str:
        """Compute time in seconds between two events per user. Returns median, mean, p90."""
        f = df[df["event_name"]==from_event][["user_uuid","event_time"]].rename(columns={"event_time":"ft"})
        t = df[df["event_name"]==to_event][["user_uuid","event_time"]].rename(columns={"event_time":"tt"})
        if f.empty or t.empty: return json.dumps({"error":"no data","n":0})
        m = f.merge(t, on="user_uuid")
        m = m[m["tt"]>m["ft"]]
        if m.empty: return json.dumps({"error":"no transitions","n":0})
        first = m.sort_values("tt").groupby("user_uuid").first().reset_index()
        first["lat"] = (first["tt"]-first["ft"]).dt.total_seconds()
        v = first[first["lat"]<3600]["lat"].values
        if len(v)==0: return json.dumps({"error":"none within 1hr","n":0})
        r = {"from":from_event,"to":to_event,"median":round(float(np.median(v)),1),
             "mean":round(float(np.mean(v)),1),"p90":round(float(np.percentile(v,90)),1),"n":len(v)}
        if "latency_pairs" not in ctx: ctx["latency_pairs"] = []
        ctx["latency_pairs"].append({**r, "values": np.clip(v,0,np.percentile(v,95)).tolist()})
        return json.dumps(r, indent=2)
    return [compute_latency]


def create_frequency_tools(df, ctx):
    @tool
    def compute_frequency_distribution(top_n: int = 20) -> str:
        """Get event frequency distribution, category breakdown, power user stats."""
        ec = df["event_name"].value_counts()
        top = [{"event":e,"count":int(c),"pct":round(c/len(df)*100,2)} for e,c in ec.head(top_n).items()]
        cat = {k:int(v) for k,v in df["category"].value_counts().items()}
        uec = df.groupby("user_uuid").size()
        pw = {"median":int(uec.median()),"p90":int(uec.quantile(0.9)),"p99":int(uec.quantile(0.99)),
              "max":int(uec.max()),"above_200":int((uec>200).sum())}
        ctx["frequency_data"] = {"top_events":top,"categories":cat,"power":pw}
        return json.dumps(ctx["frequency_data"], indent=2)
    return [compute_frequency_distribution]


def create_temporal_tools(df, ctx):
    DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    @tool
    def compute_temporal() -> str:
        """Compute day x hour temporal distribution, peak/off-peak times."""
        d = df.copy()
        d["hour"] = d["event_time"].dt.hour
        d["day"] = d["event_time"].dt.day_name()
        pivot = d.groupby(["day","hour"]).size().reset_index(name="c")
        pivot = pivot.pivot_table(index="day",columns="hour",values="c",fill_value=0)
        pivot = pivot.reindex(DAYS).reindex(columns=range(24),fill_value=0)
        hourly = d.groupby("hour").size()
        daily = d.groupby("day").size().reindex(DAYS)
        data = {"peak_hour":int(hourly.idxmax()),"off_hour":int(hourly.idxmin()),
                "peak_day":daily.idxmax(),"low_day":daily.idxmin(),
                "ratio":round(float(hourly.max()/hourly.min()),1) if hourly.min()>0 else 0}
        ctx["temporal"] = data
        ctx["temporal_matrix"] = pivot.values.tolist()
        return json.dumps(data, indent=2)
    return [compute_temporal]


def create_user_journey_tools(df, ctx):
    @tool
    def compute_user_journey_stats(min_events: int = 5) -> str:
        """Compute per-user journey statistics: completion rates, issue encounters, and user types."""
        s = df.sort_values(["user_uuid", "event_time"]).copy()
        
        # Define milestones
        milestones = {
            "search": ["bus_search", "_bus-search_user-bus-search"],
            "select": ["select_seat", "pageview_seat_selection"],
            "pay_init": ["payment_initiate", "PaymentPage_payment initiated"],
            "pay_done": ["payment_success"]
        }
        
        # Define issues
        issues = {
            "pay_fail": ["payment_failed", "paymentFailed_backpressed"],
            "app_error": ["app_error", "crash", "anr"],
            "no_results": ["bus_no_results", "zero_results"]
        }

        # Aggregation
        def analyze_user(g):
            evts = set(g["event_name"])
            res = {
                "events": len(g),
                "has_search": bool(evts & set(milestones["search"])),
                "has_select": bool(evts & set(milestones["select"])),
                "has_pay_init": bool(evts & set(milestones["pay_init"])),
                "has_pay_done": bool(evts & set(milestones["pay_done"])),
                "has_pay_fail": bool(evts & set(issues["pay_fail"])),
                "has_error": bool(evts & set(issues["app_error"])),
            }
            return pd.Series(res)

        # Filter for meaningful users
        valid_users = df.groupby("user_uuid").size()
        valid_users = valid_users[valid_users >= min_events].index
        sub = df[df["user_uuid"].isin(valid_users)]
        
        stats = sub.groupby("user_uuid").apply(analyze_user)
        total = len(stats)
        
        # Completion funnel
        funnel = {
            "searched": int(stats["has_search"].sum()),
            "selected": int(stats["has_select"].sum()),
            "pay_init": int(stats["has_pay_init"].sum()),
            "booked": int(stats["has_pay_done"].sum())
        }
        
        # User types
        types = {
            "browsers": int(stats[stats["has_search"] & ~stats["has_select"]].shape[0]),
            "shoppers": int(stats[stats["has_select"] & ~stats["has_pay_init"]].shape[0]),
            "attempters": int(stats[stats["has_pay_init"] & ~stats["has_pay_done"]].shape[0]),
            "bookers": int(stats["has_pay_done"].sum())
        }
        
        # Issues
        friction = {
            "faced_payment_failure": int(stats["has_pay_fail"].sum()),
            "faced_app_error": int(stats["has_error"].sum())
        }
        
        metrics = {
            "total_analyzed_users": total,
            "funnel_counts": funnel,
            "user_types": types,
            "friction_counts": friction,
            "conversion_rate": round(funnel["booked"]/total*100, 1) if total else 0
        }
        
        ctx["user_journey_stats"] = metrics
        return json.dumps(metrics, indent=2)

    return [compute_user_journey_stats]
