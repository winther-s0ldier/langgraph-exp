from typing import TypedDict, Annotated
import operator


def merge_dicts(a: dict, b: dict) -> dict:
    merged = a.copy()
    merged.update(b)
    return merged


class AnalyticsState(TypedDict):
    dataset_path: str
    dataset_summary: dict
    metric_results: Annotated[dict, merge_dicts]
    compiled_report: dict
    errors: Annotated[list, operator.add]
