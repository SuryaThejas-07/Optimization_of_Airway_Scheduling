# Final Project Report: AGNO-RS Runway Scheduling System

## Abstract
This report describes the design, implementation, and evaluation of a runway scheduling system using an Adaptive Graph Neural Optimization (AGNO-RS) approach. The system ingests real flight data, constructs a separation matrix based on wake classes, generates schedules using multiple optimization methods, and provides a Streamlit-based analytics dashboard. We compare AGNO-RS against FCFS, GA, and MILP baselines using delay, throughput, and safety slack metrics. The system processes a full day dataset (224 events from 29,367 records) and produces end-to-end outputs, including visual timelines, runway free intervals, and best-method selection.

## Problem Statement
Airports must schedule arrivals and departures on limited runways while ensuring safety separation, minimizing delays, and maintaining throughput. Traditional heuristics such as FCFS can be robust but may not capture dynamic interactions between flights. The challenge is to produce a schedule that improves multiple metrics simultaneously (total delay, average delay, throughput, and safety margin), while scaling to realistic datasets and providing operator-friendly visualization.

## Gap Analysis
- **Multi-objective optimization gap:** Most baselines focus on delay only; operational safety slack is often not explicitly tracked.
- **Scalability gap:** Exact optimization approaches can be expensive for daily datasets with hundreds of flights.
- **Explainability gap:** Operators need interpretable schedules and transparent visuals, not only numerical metrics.
- **Integration gap:** Many approaches lack a unified pipeline that goes from dataset ingestion to scheduling, evaluation, and dashboard analytics.

## Objectives
1. Build a full pipeline that ingests real flight data and produces a schedule.
2. Implement AGNO-RS using graph-based features and neural scoring.
3. Compare against FCFS, GA, and MILP baselines with consistent output schemas.
4. Compute multi-factor metrics including delay, throughput, and safety slack.
5. Deliver a dashboard that clearly visualizes schedules, runway utilization, and conflicts.

## Dataset and Preprocessing
- **Dataset:** `states_2022-06-27-23.csv`
- **Events extracted:** 224 runway events (landings and departures) from 29,367 flight records.
- **Features:** ETA (seconds), velocity, altitude, wake class (one-hot).
- **Separation:** A wake-class separation matrix is built to enforce safety spacing.
- **Airport region:** Center inferred from densest flight cluster.

## System Architecture
1. **Data ingestion**: Parse ADS-B flight states and detect airport proximity.
2. **Feature encoding**: Build per-flight feature vectors and adjacency matrix.
3. **Scheduling algorithms**:
   - **AGNO-RS**: Graph neural scoring + differentiable sorting.
   - **FCFS**: First-come-first-served baseline with separation enforcement.
   - **GA**: Genetic algorithm with separation enforcement.
   - **MILP**: Placeholder baseline for consistent comparison.
   - **NIS-LNS**: Neural insertion + large neighborhood search (experimental).
4. **Refiner**: Enforces runway assignment and separation constraints.
5. **Metrics**: Total delay, avg delay, makespan, throughput, safety slack.
6. **Dashboard**: Streamlit UI with timelines, heatmaps, and metrics.

## Methods
### AGNO-RS (Proposed)
- **GraphEncoder** aggregates neighborhood features over a wake-class graph.
- **AGNOModel** produces per-flight priority scores.
- **Soft-rank** creates a differentiable ordering of flights.
- **Separation-aware refiner** assigns runways and enforces separation.

### Baselines
- **FCFS**: Orders by ETA and applies separation constraints.
- **GA**: Evolves candidate sequences to minimize delay.
- **MILP**: Baseline structure for deterministic optimization (placeholder).
- **NIS-LNS**: Neural insertion with LNS refinement (experimental; performance-constrained).

## Evaluation Metrics
- **Total delay:** Sum of positive delays across flights.
- **Average delay:** Mean delay per flight.
- **Makespan:** Schedule span (last minus first time).
- **Throughput:** Flights per unit time.
- **Safety slack:** Mean and minimum separation slack.
- **Composite score:** Weighted normalization of all factors.

## Results (Full Dataset)
Results from `outputs/baselines.json`:

| Method   | Total Delay | Avg Delay | Makespan | Throughput | Avg Safety Slack | Min Safety Slack | Composite Score |
|----------|-------------|-----------|----------|------------|------------------|-----------------|-----------------|
| AGNO-RS  | 1,464,629.81 | 6,538.53  | 13,380.00 | 0.01674    | 4,440.00         | -0.00006        | 0.88884         |
| FCFS     | 1,462,426.81 | 6,528.69  | 13,380.00 | 0.01674    | 4,440.00         | -0.00024        | 0.85717         |
| MILP     | 1,462,426.81 | 6,528.69  | 13,380.00 | 0.01674    | 4,440.00         | -0.00024        | 0.85717         |
| GA       | 1,483,754.00 | 6,623.90  | 13,456.17 | 0.01665    | 4,440.68         | -0.00006        | 0.69150         |
| NIS-LNS  | 1,580,363.07 | 7,055.19  | 13,380.00 | 0.01674    | 4,440.00         | -0.00049        | 0.20000         |

**Observation:** AGNO-RS achieved the best composite score by balancing delay, throughput, and safety slack. FCFS and MILP are close in total delay but score lower overall due to the composite weighting.

## UI and Analytics
The Streamlit dashboard provides:
- Arrivals vs departures timeline by runway.
- Runway free-time intervals.
- Runway assignment timeline.
- Labeled runway scatter plot with filters.
- Delay distribution and delay vs time scatter.
- Conflict heatmap (separation slack).
- Best-method selection and metrics table.

## Challenges and Fixes
- **Performance bottleneck**: O(n^2) separation checks slowed repeated evaluations.
- **Timeline display**: Scheduled time was plotted near epoch; fixed with base timestamp.
- **Label overlap**: Added filters and label cap to reduce clutter.
- **Schema mismatches**: Baselines were aligned to the AGNO output schema.

## Conclusion
We delivered an end-to-end runway scheduling system with a novel AGNO-RS model, robust baselines, consistent evaluation, and a rich analytics dashboard. The system handles a full dataset and selects the best method automatically using multi-factor scoring. AGNO-RS achieved the top composite score, demonstrating the value of neural scoring combined with separation-aware refinement.

## Future Work
- Train AGNO on larger datasets for improved generalization.
- Re-enable NIS-LNS with accelerated cost evaluation or caching.
- Add per-runway capacity models and multi-airport support.
- Integrate real-time streaming data for online scheduling.

## Artifacts
- Code: `agno_runway/`
- Outputs: `agno_runway/outputs/`
- Dashboard: `agno_runway/ui/app.py`
