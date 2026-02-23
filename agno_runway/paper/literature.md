# Literature Review

The runway scheduling literature spans exact optimization, heuristics, and learning-driven methods. A broad survey by Ikli et al. catalogs MILP formulations, metaheuristics, and reinforcement learning variants, underscoring both the effectiveness and computational cost of exact optimization in complex, real-time settings [@ikli2021survey]. Zhou et al. propose a multi-runway genetic algorithm with sliding-window decisions, achieving competitive performance while relying on population-based search [@zhou2014ga].

AGNO-RS departs from these paradigms by representing aircraft interactions as a conflict graph and learning a differentiable ranking policy end-to-end. This shifts the primary optimization mechanism from combinatorial search to continuous, data-driven policy optimization while retaining explicit safety guarantees via constraint layers.
