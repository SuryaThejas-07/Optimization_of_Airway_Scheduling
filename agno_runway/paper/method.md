# Method

## Problem Setting
We schedule landing and takeoff events to multiple runways with separation constraints derived from wake turbulence categories. Each aircraft is modeled as a node in a dynamic graph; edges represent wake conflicts and temporal interference. Node features include ETA, velocity, altitude, and inferred wake class.

## AGNO-RS Architecture
1. **Graph Neural Encoder**: Message passing propagates local conflict context across aircraft nodes.
2. **Differentiable Ranking**: A continuous permutation (soft ranking) produces a schedule order with gradient flow.
3. **Safety Layer**: Lagrangian penalties enforce separation constraints and a robust refiner ensures operational feasibility.

## Novelty Relative to Base Papers
Unlike MILP and GA-based methods, AGNO-RS does not rely on branch-and-bound or population search. It optimizes a differentiable surrogate, enabling learning from data with gradient-based updates and direct integration of graph structure. This is fundamentally distinct from the survey and GA approaches, which do not incorporate end-to-end differentiable ordering or graph neural conflict encoding.

## Objective
We minimize total delay and makespan subject to separation constraints. A soft penalty term enforces safety while allowing training stability:

$$
\min_{\pi} \sum_i (t_i - \text{ETA}_i) + \lambda \sum_{i,j} \max(0, S_{ij} - (t_i - t_j))
$$

Where $\pi$ is the ordering policy, $t_i$ are scheduled times, and $S_{ij}$ are wake separation requirements.
