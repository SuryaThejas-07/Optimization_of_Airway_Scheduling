# Optimization of Runway Scheduling (AGNO-RS+)

AGNO-RS+ is a learning-driven runway scheduling system that ingests ADS-B state data, detects runway events, and produces safe, high-throughput schedules. It compares against FCFS, GA, and MILP baselines and visualizes outcomes in a Streamlit dashboard.

## Highlights
- End-to-end pipeline: ingest data -> detect events -> build separation matrix -> schedule -> metrics -> dashboard.
- Graph-aware scoring model with safety-aware sequencing and refinement.
- Comparative analytics across baselines with delay, throughput, and safety slack metrics.
- Critical/emergency scenario view that compares how methods handle high-priority flights.

## How It Works
1. **Ingest**: Load ADS-B state vectors and infer airport center.
2. **Detect events**: Identify takeoffs/landings in the runway geofence.
3. **Score**: Use a graph-aware model to estimate per-flight priority.
4. **Sequence**: Enforce wake-based separation and runway assignment.
5. **Refine**: Reduce delay while preserving safety slack and runway stability.
6. **Explain**: Visualize rationale, timelines, conflicts, and comparisons.

## Quick Start
1. Create and activate a virtual environment (optional)
2. Install dependencies:
   - `pip install pandas numpy torch plotly streamlit`
3. Run the pipeline and launch the UI:
   - `python -m agno_runway.main`

### CUDA / Device Selection
To force GPU usage:
- `python -m agno_runway.main --device cuda`
- `python -m agno_runway.gpu_train --device cuda`

Use `--device cpu` to force CPU.

## Inputs and Outputs
- Input dataset: `states_2022-06-27-23.csv`
- Outputs in `agno_runway/outputs/`:
  - `flights.csv`, `separation.json`, `schedule.json`, `baselines.json`, `best_schedule.json`

## Dashboard
Launches automatically from `main.py` and includes:
- Arrivals vs departures timeline
- Runway free-time intervals and utilization timeline
- Assignment rationale table
- Conflict heatmap and delay analytics
- Critical/emergency scenario comparison across methods

## Project Structure
- `agno_runway/data`: data loading, event extraction, separation builder
- `agno_runway/optimizer`: AGNO-RS+ model and baseline optimizers
- `agno_runway/analytics`: metrics and conflict checks
- `agno_runway/ui`: Streamlit dashboard
- `agno_runway/outputs`: generated artifacts

## Notes
- If airport inference fails, pass `--airport-lat` and `--airport-lon` to `main.py`.
- The UI highlights AGNO-RS+ as the designated best method and also reports the top composite score from the latest run.
