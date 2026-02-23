# AGNO-RS+: Adaptive Graph Neural Optimization for Runway Scheduling
## Final Project Report

### 1. Abstract
This report presents the research, design, and implementation of **AGNO-RS+**, an advanced runway scheduling system that leverages **Graph Neural Networks (GNN)** to optimize aircraft sequencing. By integrating neural priority scoring with a safety-aware multi-runway assignment logic, AGNO-RS+ achieves a **~14% reduction in makespan** over traditional First-Come-First-Served (FCFS) benchmarks. The system is verified on real ADS-B flight data and features a professional-grade analytics dashboard.

---

### 2. The Problem: The "Runway Bottleneck"
Global air traffic operations are currently limited by static, heuristic-based scheduling (e.g., FCFS). These methods often:
1.  Fail to account for the heterogeneous wake patterns of modern mixed-traffic fleets.
2.  Treat multiple runways as a single conceptual queue, leading to significant idle time.
3.  Lack real-time adaptive scoring for flights with varying velocities and fuel constraints.

---

### 3. The AGNO-RS+ Solution
Our approach introduces a **two-layer optimization architecture**:

#### A. Neural Prioritization Layer (AGNO-RS Model)
- **Graph Representation**: Flights are modeled as nodes. Edges represent safety separation constraints (Wake Class H/M/L).
- **GraphEncoder**: Aggregates spatial and temporal features (Altitude, Velocity, ETA) across neighboring flights.
- **Differentiable Ranking**: Uses a scoring head to produce a permutation that minimizes global delay.

#### B. Architectural Breakthrough: Multi-Runway Parallelism
Previously, controllers enforced separation across the entire airport theater. Our **Robust Multi-Runway Refiner** unlocks efficiency by:
- **Independent Lane Enforcement**: Separation is only enforced between flights sharing the same physical runway.
- **Dynamic Balancing**: Flights are assigned to the earliest available runway that satisfies their specific wake separation from the previous operation on **that specific lane**.

---

### 4. Experimental Results
Tested on a dataset of **224 runway events** extracted from **29,367 ADS-B records**.

| Method | Total Delay (s) | Avg Delay (s) | Makespan (s) | Throughput (f/s) |
| :--- | :--- | :--- | :--- | :--- |
| **AGNO-RS+** | **954,714** | **4,262** | **8,550** | **0.0262** |
| GA (Genetic) | 1,058,501 | 4,725 | 9,752 | 0.0229 |
| FCFS | 1,091,325 | 4,871 | 9,930 | 0.0225 |

**Analysis**: AGNO-RS+ significantly outperforms FCFS because it "packs" the runways more tightly. The GNN learns to prioritize aircraft that have shorter separation requirements (e.g., L following H) to maximize throughput.

---

### 5. Implementation & UX
The system includes a **professional Streamlit Dashboard** providing:
- **Arrivals/Departures Timeline**: Visual slot allocation.
- **Conflict Heatmap**: Per-runway safety verification.
- **Critical Flight Analysis**: Focused view on high-priority/emergency flights.

---

### 6. Conclusion
AGNO-RS+ demonstrates that deep learning can significantly improve safety-critical logistics. The combination of **graph-aware scoring** and **multi-runway architectural logic** represents a significant advancement over legacy ATC scheduling techniques.

---

### 7. Future Work
- Integration of **Graph Attention Networks (GAT)** for dynamic edge weighting.
- Adaptation to **Weather-Dynamic Separation** (increasing buffer during visibility loss).
- Live deployment via **OpenSky Network API** streams.
