import streamlit as st
import os
import sys
import time
import pandas as pd
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants
SUBSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_subset.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "analytics_report.html")

st.set_page_config(page_title="LangGraph Analytics", layout="wide")

st.title("LangGraph Analytics Pipeline")
st.markdown("Automated multi-agent analysis for event logs.")

if "running" not in st.session_state:
    st.session_state.running = False

def run_analysis():
    st.session_state.running = True
    try:
        from create_subset import create_subset
        from agents.graph import build_graph

        with st.status("Starting Analysis Pipeline...", expanded=True) as status:
            # 1. Prepare Data
            st.write("Preparing dataset...")
            if not os.path.exists(SUBSET_PATH):
                create_subset()
                st.write("Data subset created")
            else:
                st.write("Using existing subset")

            # 2. Build Graph
            st.write("Building agent graph...")
            graph = build_graph()
            
            initial_state = {
                "dataset_path": SUBSET_PATH,
                "dataset_summary": {},
                "metric_results": {},
                "compiled_report": {},
                "errors": [],
            }
            
            # 3. Execute Graph
            st.write("Agents analyzing data (this may take 2-3 minutes)...")
            start = time.time()
            result = graph.invoke(initial_state)
            elapsed = time.time() - start
            
            status.update(label=f"Analysis Complete ({round(elapsed, 1)}s)", state="complete", expanded=False)
            
            # 4. Show Results
            if "compiled_report" in result and "html_path" in result["compiled_report"]:
                 st.success("Report generated successfully!")
                 return True
            else:
                 errors = result.get("errors", [])
                 st.error(f"Pipeline failed with errors: {errors}")
                 return False

    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        return False
    finally:
        st.session_state.running = False



if st.button("Run Analysis", disabled=st.session_state.running, type="primary", use_container_width=True):
    if run_analysis():
        st.rerun()


st.divider()

# Report Display (Full Width)
if os.path.exists(REPORT_PATH):
    # Get last modified time
    # Get last modified time (Convert to IST)
    mtime = os.path.getmtime(REPORT_PATH)
    dt_utc = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
    ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    last_updated = dt_utc.astimezone(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')
    
    st.caption(f"Last updated: {last_updated}")
    
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_html = f.read()
    
    # Render with increased height and scrolling
    st.components.v1.html(report_html, height=1200, scrolling=True)

else:
    st.info("Click **Run Analysis** to start the pipeline.")
