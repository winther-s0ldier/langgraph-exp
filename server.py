import os
import sys
import time
import json
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SUBSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_subset.csv")
SOURCE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Commuter Users Event data.csv")

_pipeline_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "elapsed_sec": None,
    "metrics_completed": [],
    "errors": [],
    "report_path": None,
}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    yield


app = FastAPI(
    title="LangGraph Analytics API",
    description="Multi-agent analytics pipeline for commuter user event data",
    version="1.0.0",
    lifespan=lifespan,
)


class PipelineResponse(BaseModel):
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_sec: Optional[float] = None
    metrics_completed: list = []
    errors: list = []
    report_path: Optional[str] = None


def _run_pipeline():
    with _lock:
        _pipeline_state["status"] = "running"
        _pipeline_state["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _pipeline_state["completed_at"] = None
        _pipeline_state["metrics_completed"] = []
        _pipeline_state["errors"] = []

    try:
        if not os.path.exists(SUBSET_PATH):
            from create_subset import create_subset
            create_subset()

        from agents.graph import build_graph

        graph = build_graph()
        initial_state = {
            "dataset_path": SUBSET_PATH,
            "dataset_summary": {},
            "metric_results": {},
            "compiled_report": {},
            "errors": [],
        }

        start = time.time()
        result = graph.invoke(initial_state)
        elapsed = time.time() - start

        compiled = result.get("compiled_report", {})

        with _lock:
            _pipeline_state["status"] = "completed"
            _pipeline_state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _pipeline_state["elapsed_sec"] = round(elapsed, 1)
            _pipeline_state["metrics_completed"] = compiled.get("metrics_completed", [])
            _pipeline_state["errors"] = compiled.get("metrics_failed", [])
            _pipeline_state["report_path"] = compiled.get("html_path")

    except Exception as e:
        with _lock:
            _pipeline_state["status"] = "failed"
            _pipeline_state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _pipeline_state["errors"] = [str(e)]


@app.get("/", response_class=HTMLResponse)
async def root():
    report = os.path.join(OUTPUTS_DIR, "analytics_report.html")
    if os.path.exists(report):
        with open(report, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<html><body style='font-family:Inter,sans-serif;background:#09090b;color:#fafafa;display:flex;align-items:center;justify-content:center;height:100vh'>"
                "<div style='text-align:center'><h1>Analytics Report</h1><p style='color:#a1a1aa'>No report generated yet. POST to /api/run to execute the pipeline.</p></div>"
                "</body></html>"
    )


@app.post("/api/run", response_model=PipelineResponse)
async def run_pipeline():
    with _lock:
        if _pipeline_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Pipeline is already running")

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return PipelineResponse(status="started", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))


@app.get("/api/status", response_model=PipelineResponse)
async def pipeline_status():
    with _lock:
        return PipelineResponse(**_pipeline_state)


@app.get("/api/metrics")
async def list_metrics():
    with _lock:
        return JSONResponse({
            "available_metrics": [
                "funnel_analysis", "dropoff_analysis", "friction_points",
                "session_metrics", "retention_analysis", "user_segmentation",
                "conversion_rates", "time_to_action", "event_frequency", "temporal_patterns",
            ],
            "completed": _pipeline_state["metrics_completed"],
            "errors": _pipeline_state["errors"],
        })


@app.get("/api/report")
async def get_report():
    report = os.path.join(OUTPUTS_DIR, "analytics_report.html")
    if not os.path.exists(report):
        raise HTTPException(status_code=404, detail="Report not generated. POST /api/run first.")
    return FileResponse(report, media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline": _pipeline_state["status"]}
