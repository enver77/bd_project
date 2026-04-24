"""
heuristic_opt.py
────────────────
Three heuristic optimisers used in the anomaly-detection pipeline.

PSO  – Particle Swarm Optimisation
         Optimises the continuous hyperparameters of Isolation Forest
         (contamination, max_samples, max_features) using silhouette score
         as a label-free fitness proxy.

GA   – Genetic Algorithm
         Selects the optimal binary feature subset from FEATURE_NAMES.
         Chromosome = binary vector; fitness = silhouette score of the
         Isolation Forest predictions on the selected features.

SA   – Simulated Annealing
         Finds the optimal decision threshold on Isolation Forest's
         continuous anomaly scores using a domain-knowledge fitness:
         maximise sensitivity on suspicious night samples while
         minimising false positives on clearly-normal deep-night samples.

All three classes expose a single .optimize() → (best_solution, best_score).
No external optimisation libraries required — only numpy.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PSO
# ─────────────────────────────────────────────────────────────────────────────

class PSO:
    """
    Maximises fitness_fn over a bounded continuous parameter space.

    Parameters
    ----------
    fitness_fn   : callable (params: np.ndarray) → float  (higher = better)
    bounds       : list of [lo, hi] pairs, one per parameter
    n_particles  : swarm size
    n_iter       : number of iterations
    w            : inertia weight
    c1           : cognitive acceleration (personal best)
    c2           : social acceleration (global best)
    seed         : random seed
    verbose      : print progress every `verbose` iterations (0 = silent)
    """

    def __init__(
        self,
        fitness_fn,
        bounds: list,
        n_particles: int = 15,
        n_iter: int = 25,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        verbose: int = 5,
    ):
        self.fitness_fn  = fitness_fn
        self.bounds      = np.array(bounds, dtype=float)   # (n_params, 2)
        self.n_particles = n_particles
        self.n_iter      = n_iter
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.rng         = np.random.default_rng(seed)
        self.verbose     = verbose

    def optimize(self):
        lo, hi  = self.bounds[:, 0], self.bounds[:, 1]
        n_p     = self.n_particles
        n_param = len(self.bounds)

        # Initialise positions and velocities
        X = self.rng.uniform(lo, hi, (n_p, n_param))
        V = self.rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1, (n_p, n_param))

        pbest        = X.copy()
        pbest_scores = np.array([self.fitness_fn(x) for x in X])
        gbest_idx    = int(np.argmax(pbest_scores))
        gbest        = pbest[gbest_idx].copy()
        gbest_score  = pbest_scores[gbest_idx]

        for it in range(1, self.n_iter + 1):
            r1 = self.rng.random((n_p, n_param))
            r2 = self.rng.random((n_p, n_param))

            V = (self.w * V
                 + self.c1 * r1 * (pbest - X)
                 + self.c2 * r2 * (gbest - X))
            X = np.clip(X + V, lo, hi)

            scores   = np.array([self.fitness_fn(x) for x in X])
            improved = scores > pbest_scores
            pbest[improved]        = X[improved]
            pbest_scores[improved] = scores[improved]

            if pbest_scores.max() > gbest_score:
                gbest_idx   = int(np.argmax(pbest_scores))
                gbest       = pbest[gbest_idx].copy()
                gbest_score = pbest_scores[gbest_idx]

            if self.verbose and it % self.verbose == 0:
                print(f"    PSO iter {it:3d}/{self.n_iter} | "
                      f"best fitness = {gbest_score:.4f}")

        return gbest, gbest_score


# ─────────────────────────────────────────────────────────────────────────────
# GA
# ─────────────────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Maximises fitness_fn over a binary chromosome of length n_genes.

    Parameters
    ----------
    fitness_fn      : callable (chromosome: np.ndarray[int]) → float
    n_genes         : chromosome length (= number of features)
    pop_size        : population size
    n_gen           : number of generations
    crossover_rate  : probability of performing crossover
    mutation_rate   : per-gene mutation probability
    min_active      : minimum number of 1-bits required (prevents degenerate)
    seed            : random seed
    verbose         : print progress every `verbose` generations (0 = silent)
    """

    def __init__(
        self,
        fitness_fn,
        n_genes: int,
        pop_size: int = 20,
        n_gen: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        min_active: int = 2,
        seed: int = 42,
        verbose: int = 5,
    ):
        self.fitness_fn     = fitness_fn
        self.n_genes        = n_genes
        self.pop_size       = pop_size
        self.n_gen          = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.min_active     = min_active
        self.rng            = np.random.default_rng(seed)
        self.verbose        = verbose

    def _init_population(self) -> np.ndarray:
        pop = self.rng.integers(0, 2, (self.pop_size, self.n_genes))
        for i in range(self.pop_size):
            if pop[i].sum() < self.min_active:
                idx = self.rng.choice(self.n_genes, self.min_active, replace=False)
                pop[i, idx] = 1
        return pop

    def _tournament_select(self, pop: np.ndarray, scores: np.ndarray,
                            k: int = 3) -> np.ndarray:
        candidates = self.rng.choice(len(pop), k, replace=False)
        return pop[candidates[int(np.argmax(scores[candidates]))]].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        if self.rng.random() < self.crossover_rate:
            point = int(self.rng.integers(1, self.n_genes))
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        return c1, c2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        flip = self.rng.random(self.n_genes) < self.mutation_rate
        chromosome = np.where(flip, 1 - chromosome, chromosome)
        if chromosome.sum() < self.min_active:
            idx = self.rng.choice(self.n_genes, self.min_active, replace=False)
            chromosome[idx] = 1
        return chromosome

    def optimize(self):
        pop    = self._init_population()
        scores = np.array([self.fitness_fn(c) for c in pop])

        best_idx        = int(np.argmax(scores))
        best_chromosome = pop[best_idx].copy()
        best_score      = scores[best_idx]

        for gen in range(1, self.n_gen + 1):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, scores)
                p2 = self._tournament_select(pop, scores)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop    = np.array(new_pop[: self.pop_size])
            scores = np.array([self.fitness_fn(c) for c in pop])

            gen_best = int(np.argmax(scores))
            if scores[gen_best] > best_score:
                best_score      = scores[gen_best]
                best_chromosome = pop[gen_best].copy()

            if self.verbose and gen % self.verbose == 0:
                print(f"    GA  gen  {gen:3d}/{self.n_gen} | "
                      f"best fitness = {best_score:.4f}")

        return best_chromosome, best_score


# ─────────────────────────────────────────────────────────────────────────────
# SA
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedAnnealing:
    """
    Maximises fitness_fn over a scalar (the anomaly-score threshold).

    Starts from initial_state and explores the neighbourhood by adding
    a uniform random step in [-step_size, +step_size].
    Worse moves are accepted with probability exp(Δ / T).

    Parameters
    ----------
    fitness_fn     : callable (threshold: float) → float  (higher = better)
    initial_state  : starting threshold value
    T_init         : initial temperature
    T_min          : minimum temperature (stop cooling below this)
    cooling        : multiplicative cooling factor applied each iteration
    step_size      : max perturbation per step
    max_iter       : maximum number of iterations
    seed           : random seed
    """

    def __init__(
        self,
        fitness_fn,
        initial_state: float,
        T_init: float = 1.0,
        T_min: float  = 0.001,
        cooling: float = 0.95,
        step_size: float = 0.01,
        max_iter: int = 1000,
        seed: int = 42,
    ):
        self.fitness_fn    = fitness_fn
        self.state         = initial_state
        self.T_init        = T_init
        self.T_min         = T_min
        self.cooling       = cooling
        self.step_size     = step_size
        self.max_iter      = max_iter
        self.rng           = np.random.default_rng(seed)

    def optimize(self):
        state      = self.state
        score      = self.fitness_fn(state)
        best_state = state
        best_score = score
        T          = self.T_init

        for _ in range(self.max_iter):
            neighbor       = state + self.rng.uniform(-self.step_size, self.step_size)
            neighbor_score = self.fitness_fn(neighbor)
            delta          = neighbor_score - score

            if delta > 0 or self.rng.random() < np.exp(delta / max(T, 1e-10)):
                state = neighbor
                score = neighbor_score

            if score > best_score:
                best_state = state
                best_score = score

            T = max(T * self.cooling, self.T_min)

        print(f"    SA  optimal threshold = {best_state:.4f} | "
              f"fitness = {best_score:.4f}")
        return best_state, best_score
