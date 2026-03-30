"""
Integration example: Using optimized data in AGNO-RS+ main pipeline

This script shows how to integrate data optimization with the existing system.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from agno_runway.data import (
    load_states,
    detect_events,
    EventConfig,
    OptimizationPipeline,
)


def loads_optimized_dataset(
    data_path: str,
    airport_lat: Optional[float] = None,
    airport_lon: Optional[float] = None,
    radius_km: float = 30.0,
    use_optimization: bool = True,
) -> Tuple[pd.DataFrame, Optional[Tuple[float, float]], dict]:
    """
    Load ADS-B data with optional optimization pipeline.

    Parameters
    ----------
    data_path : str
        Path to CSV/Parquet file
    airport_lat, airport_lon : float, optional
        Airport coordinates
    radius_km : float
        Geofence radius
    use_optimization : bool
        Enable optimization (outlier removal, noise reduction, etc.)

    Returns
    -------
    Tuple[pd.DataFrame, Optional[Tuple[float, float]], dict]
        (events_dataframe, airport_coords, optimization_metadata)
    """

    print("Loading data...")
    states_df, airport = load_states(
        data_path,
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        radius_km=radius_km,
        use_cache=True,
        prefer_parquet=True,
    )
    print(f"✓ Loaded {len(states_df)} ADS-B records")

    metadata = {
        "source_file": str(data_path),
        "raw_records": len(states_df),
        "airport_center": airport,
    }

    if not use_optimization:
        # Skip optimization, return raw events
        if airport:
            config = EventConfig(airport_center=airport, radius_km=radius_km)
            events_df = detect_events(states_df, config)
        else:
            events_df = pd.DataFrame()
        metadata["optimization"] = "skipped"
        return events_df, airport, metadata

    # =========================================================================
    # RUN OPTIMIZATION PIPELINE
    # =========================================================================

    pipeline = OptimizationPipeline(use_cache=True)

    # 1. Clean raw data
    print("\nCleaning data...")
    states_clean = pipeline.process_raw_data(
        states_df,
        remove_outliers=True,
        smooth=True,
        outlier_contamination=0.05,
    )
    metadata["records_after_cleaning"] = len(states_clean)
    metadata["outliers_removed"] = len(states_df) - len(states_clean)

    # 2. Detect events
    if airport is None:
        print("⚠️  Could not infer airport, using raw data")
        return states_clean, airport, metadata

    print("Detecting runway events...")
    config = EventConfig(airport_center=airport, radius_km=radius_km)
    events_df = detect_events(states_clean, config)
    metadata["detected_events"] = len(events_df)

    if events_df.empty:
        print("⚠️  No events detected")
        return events_df, airport, metadata

    print(f"✓ Detected {len(events_df)} events")
    print(f"  Event types: {events_df['event_type'].value_counts().to_dict()}")

    # 3. Engineer features
    print("Engineering features...")
    engineered_df = pipeline.process_runway_events(
        events_df,
        airport_center=airport,
        engineer_features=True,
    )
    engineered_cols = [c for c in engineered_df.columns if c not in events_df.columns]
    metadata["engineered_features"] = engineered_cols

    # 4. Data stratification (for evaluation metrics)
    print("Preparing training/test split...")
    train_df, test_df, class_weights = pipeline.prepare_for_training(
        engineered_df,
        test_size=0.2,
        stratify=True,
    )
    metadata["train_samples"] = len(train_df)
    metadata["test_samples"] = len(test_df)
    metadata["class_weights"] = class_weights

    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(engineered_df)}")
    print(f"  Train: {len(train_df)} (stratified)")
    print(f"  Test:  {len(test_df)} (held-out time window)")
    print(f"Features engineered: {len(engineered_cols)}")
    if class_weights:
        print(f"Class weights: {class_weights}")

    # Return combined dataset (union of train + test)
    optimized_df = pd.concat([train_df, test_df], ignore_index=True)
    return optimized_df, airport, metadata


def save_optimization_report(
    output_dir: str,
    metadata: dict,
) -> None:
    """Save optimization metadata and statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    report_path = output_dir / "optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n📊 Report saved: {report_path}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """Example integration with AGNO-RS+ pipeline."""

    # Option 1: With optimization (recommended)
    print("\n" + "=" * 70)
    print("EXAMPLE 1: WITH OPTIMIZATION (Recommended)")
    print("=" * 70)
    events_df, airport, metadata = loads_optimized_dataset(
        "states_2022-06-27-23.csv",
        use_optimization=True,
    )
    save_optimization_report("outputs", metadata)

    # Option 2: Without optimization (faster for testing)
    print("\n" + "=" * 70)
    print("EXAMPLE 2: WITHOUT OPTIMIZATION (Baseline)")
    print("=" * 70)
    events_baseline, _, metadata_baseline = loads_optimized_dataset(
        "states_2022-06-27-23.csv",
        use_optimization=False,
    )
    print(f"Baseline events: {len(events_baseline)}")
    print(f"Optimized events: {len(events_df)}")

    # Option 3: With custom airport coordinates
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CUSTOM AIRPORT (SFO)")
    print("=" * 70)
    events_sfo, airport_sfo, metadata_sfo = loads_optimized_dataset(
        "states_2022-06-27-23.csv",
        airport_lat=37.6213,
        airport_lon=-122.3790,
        use_optimization=True,
    )
    print(f"Airport: {airport_sfo}")
    print(f"Events at SFO: {len(events_sfo)}")

    print("\n✅ All examples completed!")
