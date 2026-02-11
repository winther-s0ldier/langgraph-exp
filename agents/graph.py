from langgraph.graph import StateGraph, START, END
from agents.state import AnalyticsState
from agents.orchestrator import orchestrator_node
from agents.compiler import compiler_node
from agents.metrics.funnel_analysis import funnel_analysis_node
from agents.metrics.dropoff_analysis import dropoff_analysis_node
from agents.metrics.friction_points import friction_points_node
from agents.metrics.session_metrics import session_metrics_node
from agents.metrics.retention_analysis import retention_analysis_node
from agents.metrics.user_segmentation import user_segmentation_node
from agents.metrics.conversion_rates import conversion_rates_node
from agents.metrics.time_to_action import time_to_action_node
from agents.metrics.event_frequency import event_frequency_node
from agents.metrics.temporal_patterns import temporal_patterns_node
from agents.metrics.user_journey_insights import user_journey_insights_node


METRIC_NODES = {
    "funnel_analysis": funnel_analysis_node,
    "dropoff_analysis": dropoff_analysis_node,
    "friction_points": friction_points_node,
    "session_metrics": session_metrics_node,
    "retention_analysis": retention_analysis_node,
    "user_segmentation": user_segmentation_node,
    "conversion_rates": conversion_rates_node,
    "time_to_action": time_to_action_node,
    "event_frequency": event_frequency_node,
    "temporal_patterns": temporal_patterns_node,
    "user_journey_insights": user_journey_insights_node,
}


def build_graph() -> StateGraph:
    graph = StateGraph(AnalyticsState)
    graph.add_node("orchestrator", orchestrator_node)
    for name, func in METRIC_NODES.items():
        graph.add_node(name, func)
    graph.add_node("compiler", compiler_node)

    graph.add_edge(START, "orchestrator")
    for name in METRIC_NODES:
        graph.add_edge("orchestrator", name)
    for name in METRIC_NODES:
        graph.add_edge(name, "compiler")
    graph.add_edge("compiler", END)

    return graph.compile()
