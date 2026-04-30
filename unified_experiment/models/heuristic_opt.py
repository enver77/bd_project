"""
heuristic_opt.py
----------------
Three heuristic optimisers used in the anomaly-detection pipeline.

PSO  - Particle Swarm Optimisation
GA   - Genetic Algorithm
SA   - Simulated Annealing

All three classes expose a single .optimize() -> (best_solution, best_score).
No external optimisation libraries required -- only numpy.
"""

import numpy as np


class PSO:
    """Maximises fitness_fn over a bounded continuous parameter space."""

    def __init__(self, fitness_fn, bounds, n_particles=15, n_iter=25,
                 w=0.7, c1=1.5, c2=1.5, seed=42, verbose=5):
        self.fitness_fn  = fitness_fn
        self.bounds      = np.array(bounds, dtype=float)
        self.n_particles = n_particles
        self.n_iter      = n_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng         = np.random.default_rng(seed)
        self.verbose     = verbose

    def optimize(self):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        n_p, n_param = self.n_particles, len(self.bounds)

        X = self.rng.uniform(lo, hi, (n_p, n_param))
        V = self.rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1, (n_p, n_param))

        pbest = X.copy()
        pbest_scores = np.array([self.fitness_fn(x) for x in X])
        gbest_idx = int(np.argmax(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        for it in range(1, self.n_iter + 1):
            r1 = self.rng.random((n_p, n_param))
            r2 = self.rng.random((n_p, n_param))
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            X = np.clip(X + V, lo, hi)
            scores = np.array([self.fitness_fn(x) for x in X])
            improved = scores > pbest_scores
            pbest[improved] = X[improved]
            pbest_scores[improved] = scores[improved]
            if pbest_scores.max() > gbest_score:
                gbest_idx = int(np.argmax(pbest_scores))
                gbest = pbest[gbest_idx].copy()
                gbest_score = pbest_scores[gbest_idx]
            if self.verbose and it % self.verbose == 0:
                print(f"    PSO iter {it:3d}/{self.n_iter} | best fitness = {gbest_score:.4f}")

        return gbest, gbest_score


class GeneticAlgorithm:
    """Maximises fitness_fn over a binary chromosome of length n_genes."""

    def __init__(self, fitness_fn, n_genes, pop_size=20, n_gen=30,
                 crossover_rate=0.8, mutation_rate=0.1, min_active=2,
                 seed=42, verbose=5):
        self.fitness_fn = fitness_fn
        self.n_genes = n_genes
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_active = min_active
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _init_population(self):
        pop = self.rng.integers(0, 2, (self.pop_size, self.n_genes))
        for i in range(self.pop_size):
            if pop[i].sum() < self.min_active:
                idx = self.rng.choice(self.n_genes, self.min_active, replace=False)
                pop[i, idx] = 1
        return pop

    def _tournament_select(self, pop, scores, k=3):
        candidates = self.rng.choice(len(pop), k, replace=False)
        return pop[candidates[int(np.argmax(scores[candidates]))]].copy()

    def _crossover(self, p1, p2):
        if self.rng.random() < self.crossover_rate:
            point = int(self.rng.integers(1, self.n_genes))
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        return c1, c2

    def _mutate(self, chromosome):
        flip = self.rng.random(self.n_genes) < self.mutation_rate
        chromosome = np.where(flip, 1 - chromosome, chromosome)
        if chromosome.sum() < self.min_active:
            idx = self.rng.choice(self.n_genes, self.min_active, replace=False)
            chromosome[idx] = 1
        return chromosome

    def optimize(self):
        pop = self._init_population()
        scores = np.array([self.fitness_fn(c) for c in pop])
        best_idx = int(np.argmax(scores))
        best_chromosome = pop[best_idx].copy()
        best_score = scores[best_idx]

        for gen in range(1, self.n_gen + 1):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop = np.array(new_pop[:self.pop_size])
            scores = np.array([self.fitness_fn(c) for c in pop])
            gen_best = int(np.argmax(scores))
            if scores[gen_best] > best_score:
                best_score = scores[gen_best]
                best_chromosome = pop[gen_best].copy()
            if self.verbose and gen % self.verbose == 0:
                print(f"    GA  gen  {gen:3d}/{self.n_gen} | best fitness = {best_score:.4f}")

        return best_chromosome, best_score


class SimulatedAnnealing:
    """Maximises fitness_fn over a scalar (the anomaly-score threshold)."""

    def __init__(self, fitness_fn, initial_state, T_init=1.0, T_min=0.001,
                 cooling=0.95, step_size=0.01, max_iter=1000, seed=42):
        self.fitness_fn = fitness_fn
        self.state = initial_state
        self.T_init = T_init
        self.T_min = T_min
        self.cooling = cooling
        self.step_size = step_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def optimize(self):
        state = self.state
        score = self.fitness_fn(state)
        best_state = state
        best_score = score
        T = self.T_init

        for _ in range(self.max_iter):
            neighbor = state + self.rng.uniform(-self.step_size, self.step_size)
            neighbor_score = self.fitness_fn(neighbor)
            delta = neighbor_score - score
            if delta > 0 or self.rng.random() < np.exp(delta / max(T, 1e-10)):
                state = neighbor
                score = neighbor_score
            if score > best_score:
                best_state = state
                best_score = score
            T = max(T * self.cooling, self.T_min)

        print(f"    SA  optimal threshold = {best_state:.4f} | fitness = {best_score:.4f}")
        return best_state, best_score
