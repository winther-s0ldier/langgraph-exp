import os
import sys
import time

# Ensure this script works from any cwd by setting paths relative to itself
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # z:\data-log-cleaning


def main():
    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    os.environ["PYTHONIOENCODING"] = "utf-8"

    subset_path = os.path.join(SCRIPT_DIR, "analysis_subset.csv")

    if not os.path.exists(subset_path):
        print("=" * 60)
        print("Step 1: Creating analysis subset...")
        print("=" * 60)
        from create_subset import create_subset
        create_subset()
        print()

    if not os.path.exists(subset_path):
        print(f"ERROR: {subset_path} not found.")
        sys.exit(1)

    print("=" * 60)
    print("Step 2: Building LangGraph pipeline...")
    print("=" * 60)

    from agents.graph import build_graph
    graph = build_graph()

    initial_state = {
        "dataset_path": subset_path,
        "dataset_summary": {},
        "metric_results": {},
        "compiled_report": {},
        "errors": [],
    }

    print()
    print("=" * 60)
    print("Step 3: Executing pipeline...")
    print("=" * 60)

    start = time.time()
    result = graph.invoke(initial_state)
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)

    compiled = result.get("compiled_report", {})
    metrics = compiled.get("metrics_completed", [])
    errors = compiled.get("metrics_failed", [])
    html_path = compiled.get("html_path", "")

    print(f"\nMetrics completed: {len(metrics)}")
    for m in sorted(metrics):
        print(f"  [OK] {m}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")

    if html_path:
        abs_path = os.path.abspath(html_path)
        print(f"\n>> Open the report:")
        print(f"   {abs_path}")


if __name__ == "__main__":
    main()
