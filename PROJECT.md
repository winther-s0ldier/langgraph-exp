# LangGraph Analytics Pipeline

Multi-agent analytics system that uses LangGraph to orchestrate 11 specialized AI agents, each analyzing a different dimension of mobile app user behavior. Agents use tool-calling (ReAct pattern) with automatic LLM provider fallback (Gemini → OpenAI).

---

## Architecture

```
streamlit_app.py          ← UI entry point (Streamlit)
server.py                 ← Optional REST API entry point (FastAPI)
run_analytics.py          ← CLI entry point (headless)
create_subset.py          ← Data preparation (samples 500 users from full CSV)

agents/
├── state.py              ← Shared state schema (TypedDict)
├── graph.py              ← LangGraph DAG: orchestrator → 11 agents (parallel) → compiler
├── orchestrator.py       ← First node: loads data, builds dataset summary
├── agent_runner.py       ← ReAct agent executor with LLM provider fallback
├── llm_client.py         ← Simple LLM call wrapper (used by compiler only)
├── tools.py              ← All tool functions that agents call during analysis
├── charts.py             ← Plotly chart builders (one per metric)
├── compiler.py           ← Final node: synthesizes results into HTML report
└── metrics/              ← 11 metric agent nodes
    ├── funnel_analysis.py
    ├── dropoff_analysis.py
    ├── friction_points.py
    ├── session_metrics.py
    ├── retention_analysis.py
    ├── user_segmentation.py
    ├── conversion_rates.py
    ├── time_to_action.py
    ├── event_frequency.py
    ├── temporal_patterns.py
    └── user_journey_insights.py
```

---

## How It Works

### Pipeline Flow

1. **Orchestrator** (`orchestrator.py`) — Loads the CSV, filters to application-only events, computes a dataset summary (user count, event count, date range, top events). Broadcasts the summary to all metric agents.

2. **11 Metric Agents** (run in parallel) — Each agent:
   - Loads the CSV independently and filters to application events
   - Creates domain-specific tools from `tools.py`
   - Runs a ReAct agent loop via `agent_runner.py`: the LLM reasons about the task, calls tools to gather data, then produces HTML-formatted insights
   - Generates a Plotly chart via `charts.py`
   - Saves raw data to `outputs/json/{metric}.json`
   - Returns `{insights, fig, data, title}` into the shared state

3. **Compiler** (`compiler.py`) — Receives all metric results, asks the LLM to synthesize a cross-metric executive summary, then builds a single HTML report (`outputs/analytics_report.html`) with sidebar navigation, KPI cards, executive summary, and individual metric sections with charts.

### LLM Provider Fallback

Both `agent_runner.py` (tool-calling agents) and `llm_client.py` (compiler) implement the same fallback chain:

```
Gemini Key 1 → Gemini Key 2 → OpenAI
```

If a key hits a 429 rate limit, the system waits 2s and tries the next provider. This makes the pipeline resilient to quota exhaustion.

---

## File Reference

### Entry Points

| File | Purpose |
|---|---|
| `streamlit_app.py` | Primary UI. Shows a "Run Analysis" button, executes the pipeline, and renders the HTML report inline via `st.components.v1.html()`. |
| `server.py` | Alternative FastAPI REST API. Endpoints: `POST /api/run` (start pipeline), `GET /api/status` (check progress), `GET /api/report` (serve HTML report), `GET /` (serve report directly). |
| `run_analytics.py` | CLI runner. Executes the pipeline headlessly and prints results to stdout. Useful for cron jobs or CI. |

### Data Preparation

| File | Purpose |
|---|---|
| `create_subset.py` | Samples ~500 users from the full CSV (`Commuter Users Event data.csv`) using stratified sampling: bookers, searchers, and onboarding-only users are proportionally represented. Outputs `analysis_subset.csv`. |

### Core Pipeline

| File | Purpose |
|---|---|
| `agents/state.py` | Defines `AnalyticsState` TypedDict — the shared state that flows through the graph. Fields: `dataset_path`, `dataset_summary`, `metric_results` (merge-reduced dict), `compiled_report`, `errors` (append-reduced list). |
| `agents/graph.py` | Builds the LangGraph `StateGraph`. Wires: `START → orchestrator → [11 metric nodes in parallel] → compiler → END`. |
| `agents/orchestrator.py` | First node. Loads CSV, filters to `category == "application"`, computes summary stats. |
| `agents/compiler.py` | Final node. Calls LLM to synthesize executive insights from all metric results. Builds the consolidated HTML report with CSS, sidebar nav, KPI cards, and embedded Plotly charts. |

### Agent Infrastructure

| File | Purpose |
|---|---|
| `agents/agent_runner.py` | Generic ReAct agent executor. Takes a system prompt and tools, runs the LLM in a tool-calling loop (max 5 iterations). Handles provider fallback (Gemini keys → OpenAI) with 429 retry logic. Used by all 11 metric agents. |
| `agents/llm_client.py` | Simple single-shot LLM call (no tool-calling). Used only by the compiler to synthesize the executive summary. Same provider fallback chain. |
| `agents/tools.py` | Factory functions that create `@tool`-decorated functions for each metric domain. Each factory takes a DataFrame and a context dict, returns a list of LangChain tools. Tools include: dataset queries, funnel computation, drop-off analysis, friction detection, session stats, retention cohorts, user clustering, conversion rates, latency measurement, frequency distribution, temporal patterns, and user journey stats. |
| `agents/charts.py` | One chart-builder function per metric. Each takes a context dict (populated by tools during agent execution) and returns a Plotly figure as an HTML string. |

### Metric Agents

All 11 files in `agents/metrics/` follow the same pattern:

```python
METRIC_NAME = "..."          # Unique identifier
TITLE = "..."                # Display title for the report
SYSTEM_PROMPT = """..."""     # Domain-specific prompt with tool-calling instructions

def {metric}_node(state):
    df = load_and_filter_csv()
    tools = create_{metric}_tools(df, ctx)
    insights, iters = run_agent(SYSTEM_PROMPT, tools)
    chart_html = build_{metric}_chart(ctx)
    save_json_output()
    return {"metric_results": {METRIC_NAME: result}}
```

| Metric | What It Analyzes |
|---|---|
| `funnel_analysis` | 8-stage booking funnel conversion (app start → payment success) |
| `dropoff_analysis` | Waterfall of user losses between funnel stages |
| `friction_points` | Events repeated within sessions (indicators of user struggle) |
| `session_metrics` | Session count, duration, bounce rate, depth per user |
| `retention_analysis` | Weekly cohort retention matrix |
| `user_segmentation` | DBSCAN clustering on behavioral features |
| `conversion_rates` | Overall conversion, payment success rate, push notification conversion |
| `time_to_action` | Latency between key event pairs (search→result, result→booking) |
| `event_frequency` | Event distribution, power user identification |
| `temporal_patterns` | Day × hour usage heatmap, peak/off-peak analysis |
| `user_journey_insights` | Per-user journey profiling: completers vs droppers vs single-session |

### Configuration

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Streamlit theme (light mode) and server settings. |
| `.streamlit/secrets.toml` | API keys (GEMINI_API_KEY, GEMINI_API_KEY_2, OPENAI_API_KEY). **Not committed to git.** |
| `.env` | Alternative key storage via environment variables (loaded by `python-dotenv`). **Not committed to git.** |
| `.gitignore` | Excludes secrets, data files, outputs, and Python artifacts from version control. |
| `requirements.txt` | Python dependencies. |

### Outputs (Generated at Runtime)

| Path | Purpose |
|---|---|
| `analysis_subset.csv` | Sampled dataset (generated by `create_subset.py`). |
| `outputs/analytics_report.html` | Final consolidated HTML report. |
| `outputs/json/{metric}.json` | Raw JSON data from each metric agent. |

---

## Deployment

1. Copy the `lang-graph-experiment/` folder to a new location.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Add your API keys to `.streamlit/secrets.toml` (at least one Gemini key required).
4. Place your data CSV one level up as `Commuter Users Event data.csv`, or pre-generate `analysis_subset.csv`.
5. Run:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Key Design Decisions

- **Parallel execution**: All 11 metric agents run concurrently via LangGraph's fan-out pattern, reducing total pipeline time from ~30min (sequential) to ~3min.
- **Tool-calling agents**: Each metric uses the ReAct pattern — the LLM decides which tools to call and in what order, making analysis adaptive rather than hardcoded.
- **Provider fallback**: Dual Gemini keys + OpenAI fallback ensures the pipeline completes even under API rate limits.
- **JSON persistence**: Raw metric data is saved as JSON so the report can be regenerated without re-running agents.
- **HTML report**: Self-contained single-file report with embedded CSS, Plotly charts, and sidebar navigation. No external dependencies at view time.
