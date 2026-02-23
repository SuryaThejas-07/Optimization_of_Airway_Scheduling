# Introduction

Runway scheduling is a safety-critical, high-impact decision problem in air traffic management. Traditional approaches prioritize deterministic optimization (e.g., mixed-integer programming) or evolutionary search (e.g., genetic algorithms), but these methods struggle with real-time adaptation and data-driven prioritization. This work introduces AGNO-RS, a fully differentiable, graph-based scheduler that learns ordering policies from operational trajectories while explicitly enforcing safety separations.

Key contributions:
- Dynamic conflict graph modeling using ADS-B states.
- Differentiable neural scheduling with a constraint-aware safety layer.
- Robust refinement for operational feasibility.
- Empirical evaluation on OpenSky-derived events with baseline comparisons.
