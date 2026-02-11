import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV = os.path.join(SCRIPT_DIR, "events.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "analysis_subset.csv")
TARGET_USERS = 500
RANDOM_SEED = 42

BOOKING_EVENTS = {
    "payment_success", "booking_confirmed", "payment_initiate",
    "ticket_confirmed", "booking_complete", "payment_completed",
    "book_ticket", "Booking_to_Ticket"
}
SEARCH_EVENTS = {
    "bus_search", "bus_result", "select_seat", "bus_detail",
    "_bus-search_user-bus-search", "_bus-search_list",
    "_location_elastic-town-search", "_location_special-town-search"
}


def create_subset():
    print(f"Loading {SOURCE_CSV}...")
    df = pd.read_csv(SOURCE_CSV)
    print(f"  Total events: {len(df):,}  |  Users: {df['user_uuid'].nunique():,}  |  Event types: {df['event_name'].nunique()}")

    user_events = df.groupby("user_uuid")["event_name"].apply(set)
    bookers = [u for u, e in user_events.items() if e & BOOKING_EVENTS]
    searchers = [u for u, e in user_events.items() if (e & SEARCH_EVENTS) and not (e & BOOKING_EVENTS)]
    onboarding = [u for u, e in user_events.items() if not (e & SEARCH_EVENTS) and not (e & BOOKING_EVENTS)]

    total = len(bookers) + len(searchers) + len(onboarding)
    rng = np.random.RandomState(RANDOM_SEED)

    n_b = max(1, int(TARGET_USERS * len(bookers) / total))
    n_s = max(1, int(TARGET_USERS * len(searchers) / total))
    n_o = TARGET_USERS - n_b - n_s

    selected = set(
        list(rng.choice(bookers, min(n_b, len(bookers)), replace=False)) +
        list(rng.choice(searchers, min(n_s, len(searchers)), replace=False)) +
        list(rng.choice(onboarding, min(n_o, len(onboarding)), replace=False))
    )

    subset = df[df["user_uuid"].isin(selected)].sort_values(["user_uuid", "event_time"]).reset_index(drop=True)
    subset.to_csv(OUTPUT_CSV, index=False)

    coverage = subset["event_name"].nunique() / df["event_name"].nunique() * 100
    print(f"  Subset: {len(subset):,} events, {subset['user_uuid'].nunique()} users, {coverage:.0f}% event coverage")
    print(f"  Saved to {OUTPUT_CSV}")
    return subset


if __name__ == "__main__":
    create_subset()
