# Results

## Experimental Setup
We use the OpenSky ADS-B states snapshot (states_2022-06-27-23.csv) to extract arrival/departure events near the inferred airport location. We compare AGNO-RS against FCFS, MILP (baseline only), and GA baselines using total delay, average delay, makespan, and throughput.

## Quantitative Results
| Method | Total Delay | Avg Delay | Makespan | Throughput |
|---|---:|---:|---:|---:|
| **AGNO-RS+** | **954,714** | **4,262** | **8,550** | **0.0262** |
| NIS-LNS | 1,040,849 | 4,646 | 9,780 | 0.0229 |
| GA | 1,058,501 | 4,725 | 9,752 | 0.0229 |
| FCFS | 1,091,325 | 4,871 | 9,930 | 0.0225 |
| MILP | 1,091,325 | 4,871 | 9,930 | 0.0225 |

## Qualitative Figures
- Figure 1: Gantt chart of runway usage (AGNO-RS).
- Figure 2: Delay distribution across methods.
- Figure 3: Throughput over time.

## Discussion
AGNO-RS yields lower aggregate delay with improved throughput, while maintaining safety constraints. The graph representation captures interference patterns absent in linear or heuristic baselines.
