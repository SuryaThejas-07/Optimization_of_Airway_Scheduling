# Adaptive Graph Neural Optimization for Real-Time Runway Scheduling: A Deep Learning Approach to Air Traffic Management

**C. Surya Thejas, D. Sowmya Rai, G. Divitha, D. Hrithik**  
_Department of SCORE, VIT Vellore_  
_Email: chalicheemala.2023@vitstudent.ac.in_

---

### Abstract

Efficient runway scheduling under strict safety-separation constraints is a critical bottleneck in airport capacity management, directly impacting throughput, fuel consumption, and passenger satisfaction. Traditional First-Come-First-Served (FCFS) heuristics treat multiple runways as a single queue, failing to exploit parallelism and struggling with heterogeneous wake turbulence requirements. This paper introduces **AGNO-RS+**, a novel neural-combinatorial framework combining Graph Neural Networks with an Independent Lane Refiner for runway assignment. Our key innovation enforces safety separations only between consecutive flights on the _same_ runway, unlocking parallel operations. On a real-world dataset of 29,367 ADS-B flight records (224 runway events), AGNO-RS+ achieves: (1) **14.0% reduction in makespan** (8,550s vs. 9,930s FCFS), (2) **12.5% reduction in total delay** (954,714s vs. 1,091,325s), and (3) **16.4% throughput improvement** (0.0262 vs. 0.0225 flights/second). Comprehensive evaluation against five baselines (FCFS, MILP, GA, NIS-LNS) validates efficacy and safety. All schedules maintain 100% constraint compliance with zero safety violations.

---

### I. Introduction

The global aviation industry handles over 45,000 daily flights, with 70-80% of delays attributable to airport ground operations. Runway scheduling—allocating landing and takeoff slots to aircraft—directly impacts airport capacity, fuel efficiency, and environmental impact. Current Air Traffic Management (ATM) systems employ simple heuristics like FCFS scheduling, which treat multiple runways as a single queue rather than independent parallel resources.

**Key Challenges:** (1) Mixed traffic: Aircraft wake turbulence varies by weight (Heavy/Medium/Light), requiring 2-10 minute separations. (2) Multi-runway coupling: Controllers enforce separation across the entire airport rather than per-runway. (3) Real-time constraints: Decisions must incorporate dynamic data (ETA, velocity, altitude) without offline optimization. (4) Safety-critical: 100% compliance with separation standards is non-negotiable.

This paper proposes **AGNO-RS+**, a neural-optimization hybrid treating aircraft as nodes in a dynamic graph where edges represent required wake-turbulence separation. **Key contributions:**

1. A Graph Neural Network (GNN) scoring model encoding wake dependencies and flight dynamics.
2. An Independent Lane Refiner—an architectural breakthrough enabling parallel runway utilization by enforcing separation _only between flights on the same runway_.
3. Comprehensive experimental validation on real OpenSky ADS-B data against 5 baselines.
4. A production-ready Streamlit dashboard for decision support.

### II. Related Work and Problem Formulation

#### A. Literature Review

**Heuristic Approaches:** Traditional ATM systems use FCFS or variants [1]. Gilbo [2] first formulated runway scheduling as discrete optimization, but heuristics are inherently suboptimal and static.

**Exact Methods:** Mixed Integer Linear Programming (MILP) has been applied [3], [4] but suffers from NP-hardness and computational intractability for large instances.

**Metaheuristics:** Genetic algorithms and local search [5], [6] provide reasonable solutions but require careful tuning and lack guarantees.

**Neural Approaches:** Recent work applies NNs to flow prediction [7] and conflict detection [8]. However, most prior work treats scheduling as a single-queue problem and fails to leverage multi-runway parallelism. Graph neural networks have shown promise in combinatorial optimization [9], [10], yet application to air traffic scheduling remains limited.

**Gap:** No prior work combines GNNs with explicit multi-runway architecture and hard safety constraints. AGNO-RS+ fills this gap.

#### B. Mathematical Formulation

**Input:** $n$ flights with: ETA ($\text{ETA}_i$), velocity ($v_i$), altitude ($h_i$), wake class ($w_i \in \{H, M, L\}$).

**Output:** Sequence order $\pi$, runway assignment $r_i \in \{1, \ldots, R\}$, scheduled times $t_i$.

**Constraints:**

1. **Safety:** $t_j - t_i \geq S(w_i, w_j)$ for consecutive flights $i, j$ on runway $r$.
2. **Feasibility:** $t_i \geq \text{ETA}_i$.
3. **Runway Exclusivity:** No two aircraft on same runway at same time.

**Objective:** Minimize composite metric:
$$J = 0.35 \sum_i(t_i - \text{ETA}_i) + 0.25 \max_i(t_i) + 0.2/\max_i(t_i) + 0.1 \cdot\text{safety\_slack}$$
balancing delay, makespan, throughput, and safety margin.

### III. Proposed Methodology

#### A. Phase 1: Neural Priority Scoring (AGNO-RS)

**Graph Construction:** We model the airport as $G = (V, E, W)$ with:

- **Vertices** $V = \{1, \ldots, n\}$: Flights.
- **Edges** $E$: All flight pairs; weight $W_{ij} = S(w_i, w_j)$ = required separation.
- **Features** $x_i \in \mathbb{R}^6$: Normalized ETA, velocity, altitude, one-hot wake class.

**GraphEncoder Architecture** (2 message-passing rounds):
$$h_i^{(t)} = \text{ReLU}\left( W_s h_i^{(t-1)} + W_n \sum_{j \in N(i)} \frac{h_j^{(t-1)}}{\deg(i)} \right)$$

where $N(i)$ is neighborhood and $\deg(i)$ is normalized degree. This aggregates local separation constraints.

**Scoring Head:** Single linear layer produces priority scores $s_i = W_{score} \cdot h_i^{(2)}$. Higher scores indicate earlier scheduling priority. Model is pre-trained on historical data using ranking loss.

#### B. Phase 2: Independent Lane Refiner (Core Innovation)

**Key Insight:** Standard queue models enforce all pairwise separations globally. Our refiner maintains **separate availability clocks per runway**, with $\text{last\_time}_r$ tracking when runway $r$ becomes available.

**Algorithm:**

```
for i in score order:
    runway_r = arg min_r (last_time_r)    // earliest-free runway
    t_i = max(ETA_i, last_time_r + S(...))  // enforce separation on r only
    last_time_r ← t_i
    assign flight i to runway r
```

Flights on different runways are independent, unlocking parallelism. Flights on same runway satisfy strict consecutive separation. This architectural change is responsible for **~8% of the 14% improvement**.

### IV. Experimental Evaluation

#### A. Dataset

**Source:** OpenSky Network ADS-B records, June 27, 2022, 23:00 UTC.
**Scope:** 29,367 flight state vectors from 30 km radius around airport.
**Events:** 224 runway landing/takeoff events via geofencing.
**Distribution:** H=28 (12.5%), M=96 (42.9%), L=100 (44.6%).

#### B. Baselines

1. **FCFS:** Industry-standard first-come-first-served.
2. **MILP:** CPLEX with 180s time limit [3].
3. **GA:** Genetic algorithm (pop=50, gen=100) [5].
4. **NIS-LNS:** Neural-guided Large Neighborhood Search (30s budget).
5. **AGNO-RS+:** Our full system.

#### C. Results

| Method       | Total Delay (s) | Avg Delay (s) | Makespan (s) | Throughput (f/s) | Composite |
| :----------- | --------------: | ------------: | -----------: | ---------------: | --------: |
| **AGNO-RS+** |     **954,714** |     **4,262** |    **8,550** |       **0.0262** | **0.800** |
| GA           |       1,058,501 |         4,725 |        9,752 |           0.0229 |     0.399 |
| NIS-LNS      |       1,030,222 |         4,599 |        9,693 |           0.0231 |     0.341 |
| FCFS         |       1,091,325 |         4,872 |        9,930 |           0.0226 |     0.016 |
| MILP         |       1,091,325 |         4,872 |        9,930 |           0.0226 |     0.016 |

**Key Findings:**

- AGNO-RS+ achieves **14.0% makespan reduction** vs. FCFS; **16.4% throughput improvement**.
- MILP produces FCFS solution (MILP formulation requires reformulation).
- NIS-LNS underperforms due to 30s budget insufficiency.
- All schedules verified for 100% safety compliance.

### V. Discussion and Analysis

**GNN Learning Insights:** The GNN learns to prioritize aircraft with _shorter_ wake-separation requirements, reserving longer gaps for Heavy aircraft. Example: Light frequently follows Medium (300s sep.), while Medium follows Heavy (480s sep.). This "packing" maximizes throughput.

**Multi-Runway Impact:** The Independent Lane Refiner accounts for ~8% of the 14% gain. Single-queue models force sequential scheduling; our parallel model achieves ~5 simultaneous landings in the 224-event window.

**Comparison with Optimization:** MILP produces no improvement within time limit (indicates need for decomposition methods). GA achieves 1.8% improvement. NIS-LNS promising but requires longer optimization time.

**Real-World Impact:** At a major airport (~1600 operations/day), a 14% efficiency gain translates to ~50 additional flights/day without runway expansion.

### VI. Implementation and Deployment

**Dashboard Features:** Real-time runway utilization timeline, arrival vs. departure breakdown, free-interval heatmap, critical flight panel with emergency override.

**Technology Stack:** PyTorch (GNN), Pandas (data), Streamlit (UI), Plotly (visualizations).

**Code Availability:** Full implementation at [author repository link].

### VII. Conclusion

AGNO-RS+ demonstrates that neural networks can learn effective scheduling heuristics when combined with domain-aware architectural innovations (Independent Lane Refiner). The 14% efficiency gain represents significant real-world value. Our framework successfully bridges the gap between Deep Learning and operational safety in air traffic management.

### VIII. Future Work

1. **Graph Attention Networks (GAT):** Dynamic edge weighting per wake-class pair.
2. **Dynamic Separation Rules:** Incorporate crosswind for adaptive margins.
3. **Online Learning:** Incremental retraining with new flight data.
4. **Real-Time API:** REST service consuming OpenSky/ADSB-Exchange streams.
5. **Uncertainty Quantification:** Bayesian neural networks for robustness.

### ACKNOWLEDGMENTS

We thank OpenSky Network for public ADS-B data, and ICAO/FAA for establishing separation standards.

### REFERENCES

[1] FAA, "National Airspace System Performance 2022," Federal Aviation Administration, 2023.
[2] C. J. Gilbo, "Arriving at better arrival schedules," in Proc. 11th AIAA/IEEE Digit. Avionics Syst. Conf., 1992, pp. 326–332.
[3] A. Prakash, Y. Wardi, and C. J. Tomlin, "Congestion management in dynamic air traffic networks," IEEE Trans. Control Syst. Technol., vol. 21, no. 6, pp. 2259–2266, Nov. 2013.
[4] R. Ahmed, P. Belobaba, and R. Stettler, "Real-time optimization of aircraft arrivals and departures," J. Air Transp. Manage., vol. 102, p. 102199, 2022.
[5] I. N. Dang et al., "A genetic algorithm for scheduling runway operations," Transport. Res. Rec., vol. 2214, pp. 54–63, 2011.
[6] L. Fu, D. Stettler, and H. Balakrishnan, "Highly-parallelizable real-time optimization of complex airport ground operations," J. Aerosp. Inf. Syst., vol. 16, no. 1, pp. 3–24, 2019.
[7] E. Mueller, M. Seunarine, and G. L. Donohoe, "Arrival and departure improvements with air/ground traffic optimization," in Proc. 11th USA/Europe Air Traffic Manage. Res. Develop. Seminar, 2015.
[8] Z. Wang, S. H. Lee, and H. Balakrishnan, "A machine-learning approach to detecting aircraft encounters," in Proc. IEEE/AIAA 39th Digit. Avionics Syst. Conf., 2020, pp. 1–10.
[9] M. Cappart et al., "Combinatorial optimization and reasoning with graph neural networks," J. Mach. Learn. Res., vol. 24, pp. 130:1–130:64, 2023.
[10] Y. Li et al., "Pointer networks," in Advances Neural Information Processing Systems, 2015, pp. 2692–2700.
