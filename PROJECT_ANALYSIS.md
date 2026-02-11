# Project Analysis: LangGraph Analytics Pipeline

## 1. System Overview
This project is a **multi-agent data analytics system** built with **LangGraph**, **Streamlit**, and **Pandas**. It automates the process of analyzing raw event logs (CSV) to generate a comprehensive, interactive HTML report.

### Architecture Diagram
```mermaid
graph TD
    User[User (Streamlit UI)] -->|Run Analysis| App[streamlit_app.py]
    App -->|Invoke| Graph[LangGraph (agents/graph.py)]
    
    subgraph "Agent Graph"
        Orchestrator[Orchestrator Node] -->|Fan Out| Agents
        
        subgraph "Metric Agents (Parallel)"
            Session[Session Metrics]
            Funnel[Funnel Analysis]
            Conversion[Conversion Rates]
            Retention[Retention Analysis]
            ...[Other Agents]
        end
        
        Agents -->|Results| Compiler[Compiler Node]
    end
    
    Compiler -->|HTML Report| App
    App -->|Display| User
```

## 2. Core Components Analysis

### [streamlit_app.py](streamlit_app.py)
**Role:** Frontend & Application Entry Point.
- **State Management:** Uses `st.session_state` to track if the analysis is running.
- **Data Prep:** Checks for `analysis_subset.csv`. If missing, calls `create_subset.py` (though currently mapped to `events.csv` via git tracking).
- **Graph Execution:** Initializes the LangGraph graph (`build_graph()`) and invokes it with the dataset path.
- **Report Display:** Checks for `outputs/analytics_report.html`, converts its "Last Updated" timestamp to **IST (UTC+5:30)**, and embeds it using `st.components.v1.html`.

### [agents/graph.py](agents/graph.py)
**Role:** Workflow Definition.
- **State:** Uses `AnalyticsState` (TypedDict) to share data (dataset path, summaries, results, errors) between nodes.
- **Structure:**
  - **Start** -> `orchestrator`
  - `orchestrator` -> **Conditional Edge** (fans out to all enabled metric agents like `session_metrics`, `funnel_analysis`, etc.).
  - Metric Agents -> `compiler`
  - `compiler` -> **End**

### [agents/orchestrator.py](agents/orchestrator.py)
**Role:** Data Validation & Global Context.
- **Logic:**
  - Reads the CSV.
  - formatting `event_time` to datetime.
  - Calculates global stats: Total Users, Total Events, Date Range.
  - **Feature:** Calculates **"Peak Day"** and **"Days Covered"** for the summary.
  - Passes this technical summary to the shared state so other agents don't need to re-calculate basics.

### [agents/agent_runner.py](agents/agent_runner.py)
**Role:** The "Brain" execution engine.
- **Function:** `run_agent(system_prompt, tools)`
- **Logic:**
  - Initializes the LLM (Gemini or OpenAI via `llm_client.py`).
  - Binds the provided Python tools (from `tools.py`) to the LLM.
  - Runs a ReAct loop:
    1.  LLM thinks and calls a tool.
    2.  Tool executes (pandas logic).
    3.  Output goes back to LLM.
    4.  LLM generates final insights.

### [agents/tools.py](agents/tools.py)
**Role:** Tool Definitions (The "Hands").
- **Content:** A library of `@tool` decorated functions using Pandas.
- **Key Tools:**
  - `compute_session_stats`: Calculates duration, bounce rate, depth.
  - `compute_funnel`: Counts users at each stage of a defined funnel.
  - `compute_retention`: Generates a cohort matrix.
  - `compute_temporal`: Heatmap of activity (Day x Hour).
- **Design:** These processing functions return JSON strings, which the LLM interprets to write insights.

### [agents/compiler.py](agents/compiler.py)
**Role:** Report Generation.
- **Logic:**
  - Aggregates `metric_results` from the state.
  - Uses `llm_client` to generate an "Executive Summary" based on all findings.
  - **Template:** Embeds everything into a responsive HTML template.
  - **Visuals:** Renders Plotly charts (as HTML snippets) and the KPI grid.

## 3. Metric Agents (Deep Dive)
Each agent in `agents/metrics/` follows a similar pattern but focuses on a specific domain.

### [metrics/session_metrics.py](agents/metrics/session_metrics.py)
**Analysis:**
- **Goal:** Analyze user session quality.
- **Process:** Defines a session (inactivity period logic), calculates metrics.
- **Trend Analysis:** explicitly splits the dataset into **T1 (First Half)** and **T2 (Second Half)** to compare stats (Duration, Bounce Rate) and identify trends.

### [metrics/conversion_rates.py](agents/metrics/conversion_rates.py)
**Analysis:**
- **Goal:** Monetization analysis.
- **Trend Analysis:** Compares **Conversion Rate** and **Payment Success Rate** between T1 and T2 to spot payment gateway issues or improvements.

### [metrics/funnel_analysis.py](agents/metrics/funnel_analysis.py)
**Analysis:**
- **Goal:** User journey bottlenecks.
- **Tools:** Uses `compute_funnel` with precise event names mappings (e.g., `bus_search` -> `select_seat`).
- **Trend Analysis:** Tracks the "Look-to-Book" ratio over time.

## 4. Deployment Strategy
- **Secrets:** Uses logic in `llm_client.py` to prioritize `st.secrets` (Streamlit Cloud) -> `secrets.toml` (Local) -> `os.environ` (Docker/System).
- **Data:** `events.csv` is tracked in git (via `.gitignore` whitelist) to ensure the cloud instance has data.
- **Dependencies:** `requirements.txt` pins `streamlit>=1.40.0` to avoid conflicts with modern `altair` versions.

## 5. Data Flow Summary
1.  **User** clicks "Run".
2.  **App** loads `events.csv`.
3.  **Orchestrator** profiles the data.
4.  **Agents** (Session, Funnel, etc.) come alive, query the data using Python tools, and form insights.
5.  **Compiler** gathers these insights + charts.
6.  **Report** (HTML) is saved to `outputs/` and displayed in the app.
