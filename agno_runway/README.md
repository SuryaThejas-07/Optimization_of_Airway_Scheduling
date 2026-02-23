# AGNO-RS: Adaptive Graph Neural Optimization for Real-Time Runway Scheduling

This project builds a learning-driven, differentiable runway scheduler from OpenSky ADS-B states data. It avoids MILP and GA as primary optimizers and focuses on a graph neural policy with explicit safety constraints.

## Folder Structure
- data/: dataset processing, event extraction, separation matrix builder
- optimizer/: GNN scheduler, differentiable ranking, safety constraints, baselines
- analytics/: metrics, conflict checks, plots
- ui/: Streamlit dashboard
- paper/: manuscript sections and references

## Requirements
- Python 3.9+
- pandas, numpy, torch, plotly, streamlit

## Quick Start
1. Install dependencies
   - pip install pandas numpy torch plotly streamlit
2. Run the pipeline
   - python .\agno_runway\main.py
3. Launch dashboard (auto in main)
   - streamlit run .\agno_runway\ui\app.py

## Outputs
- agno_runway/outputs/flights.csv
- agno_runway/outputs/separation.json
- agno_runway/outputs/schedule.json
- agno_runway/outputs/baselines.json

## Notes
- Set airport coordinates with --airport-lat and --airport-lon if auto-inference is unreliable.
- GPU is used automatically when CUDA is available.
