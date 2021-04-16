import argparse
from dataclasses import dataclass
import json
from math import floor, sqrt

import numpy as np


# Problem definition and optimization convergence criterion
@dataclass
class Constants:
  N: int
  convergence_fitness: float
  convergence_evals: int


# Invariant selection parameters for the evolutionary strategy
@dataclass
class Selection:
  lamda: int
  mu: int
  weights: np.ndarray
  mu_eff: float


# Invariant adaptation parameters for the evolutionary strategy
@dataclass
class Adaptation:
  c_cov: float
  c_sigma: float
  c_1: float
  c_mu: float
  chi_N: float
  damp: float
  decompose_threshold: float


# Simple class for tracking the state of the algorithm at a given iteration
@dataclass
class State:
  sigma: float
  means: np.ndarray
  means_prime: np.ndarray
  p_cov: np.ndarray
  p_sigma: np.ndarray
  B: np.ndarray
  D: np.ndarray
  C: np.ndarray
  C_inv_sqrt: np.ndarray
  eigenevals: int

  def generation_tuple(self):
    '''Returns the necessary state attributes for generating new solutions'''
    return (self.sigma, self.means, self.B, self.D)


# Simple class for holding optimization results
@dataclass
class Results:
  solution: np.ndarray
  fitness: float


# global variables for some of the read only dataclasses
constants = None
selection = None
adaptation = None


def shifted_rastrigin(x, y):
  ''' Shifted rastrigin function to prevent zero initialization from solving the
  function landscape trivially

  Args:
    x:  an array of x values
    y:  and array of y values

  Returns:
    a floating point fitness value for values x and y
  '''
  return ((x-5)**2 - 10 * np.cos(2 * np.pi + (x-5))) + \
         ((y-5)**2 - 10 * np.cos(2 * np.pi + (y-5))) + 20


def get_fitnesses(pool):
  '''Vectorized helper function for calculating the fitnesses of a pool of
  candidate solutions.

  Args:
    pool:  an array of potential solutions which minimize the fitness function

  Returns:
    an array of fitness values where the ith fitness corresponds to the
    candidate solution in column i
  '''
  return shifted_rastrigin(pool[0], pool[1])


def load_config():
  '''Helper for loading the JSON configuration

  Returns:
    a de-serialized dictionary of the JSON config
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      type=str,
                      required=True,
                      help='Path to the JSON config file.')
  args = parser.parse_args()
  return json.load(open(args.config))


def init(config):
  '''Performs initial setup reslting in the following:

  (1) dimensionality and convergence criterion are set from the config file
  (2) selection parameters are calculated based on best practices given the
      dimensionality of the search space
  (3) adaptation parameters are calculated similarly
  (4) the initial state is populated (either randomly or via a random seed)

  Args:
    config:  the JSON config dictionary

  Returns:
    an initial state for the optimization
  '''
  global constants, selection, adaptation

  required = {'N', 'sigma', 'convergence_fitness', 'convergence_eval_factor'}
  given = set(config.keys())
  if not required <= given:
    raise ValueError(
        f'Required keys are missing from config: {required - given}')

  # constants
  N = config['N']
  convergence_fitness = config['convergence_fitness']
  factor = config['convergence_eval_factor']
  convergence_evals = factor * N**2
  constants = Constants(N, convergence_fitness, convergence_evals)

  # selection
  lamda = 4 + floor(3 * np.log(N))
  mu = lamda / 2
  weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
  mu = floor(mu)
  weights = weights / weights.sum()
  mu_eff = 1 / np.sum(weights**2)
  selection = Selection(lamda, mu, weights, mu_eff)

  # adaptation
  c_cov = (4 + mu_eff/N) / (N + 4 + 2*mu_eff/N)
  c_sigma = (mu_eff+2) / (N+mu_eff+5)
  c_1 = 2 / ((N + 1.3)**2 + mu_eff)
  c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)**2 + mu_eff))
  chi_N = N**0.5 * (1 - 1 / (4*N) + 1 / (21 * N**2))
  damp = 1 + 2 * max(0, sqrt((mu_eff-1) / (N+1)) - 1) + c_sigma
  decompose_threshold = lamda / (c_1+c_mu) / N / 10
  adaptation = Adaptation(c_cov, c_sigma, c_1, c_mu, chi_N, damp,
                          decompose_threshold)

  # initial state
  if "seed" in config:
    np.random.seed(config['seed'])
  sigma = config['sigma']
  means = np.random.rand(N, 1)
  p_cov = np.zeros((N, 1))
  p_sigma = np.zeros((N, 1))
  B = np.identity(N)
  D = np.ones(N)
  C = B @ np.diag(D**2) @ B.T
  C_inv_sqrt = B @ np.diag(D**(-1)) @ B.T
  state = State(sigma, means, None, p_cov, p_sigma, B, D, C, C_inv_sqrt, 0)
  return state


def update_means(pool, state):
  '''Updates the means and means_prime for each decision variable.

  Args:
    pool:  an ndarray of candidate solutions
    state:  the current state of the optimization variables

  Returns:
    the state with updated means and means_prime
  '''
  state.means_prime = state.means
  state.means = pool[:, :selection.mu] @ selection.weights.reshape(-1, 1)
  return state


def update_p_sigma(state):
  '''Updates p_sigma for the current state.

  Args:
    state:  the current state of the optimization variables

  Returns:
    the state with updated p_sigma
  '''
  c_sigma = adaptation.c_sigma
  discount = 1 - c_sigma
  complements = sqrt(c_sigma * (2-c_sigma) * selection.mu_eff)
  normal = state.C_inv_sqrt @ (state.means - state.means_prime) / state.sigma
  state.p_sigma = discount * state.p_sigma + complements*normal
  return state


def get_indicator(evals, state):
  '''Indicator function for compensating for small variance loss.

  Args:
    evals:  the number of solutions evaluated so far
    state:  the current state of the optimization variables

  Returns:
    a boolean indicator

  NOTE: The use of this indicator is two fold. In the first case, this boolean
  is taken as is (which is usually True) which causes p_cov to tend to be
  sampled from the normal distribution N(0, C_k+1) where C is the covariance
  matrix. In the second case, this boolean is inverted (causing it to typically
  be False) and indicates that mall corrections due to variance loss should be
  made.
  '''
  norm = np.linalg.norm(state.p_sigma)
  root = sqrt(1 - (1 - adaptation.c_sigma)**(2 * evals / selection.lamda))
  lhs = norm / root / adaptation.chi_N
  rhs = 1.4 + 2 / (constants.N + 1)
  return lhs < rhs


def update_p_cov(indicator, state):
  '''Updates p_cov for the current state.

  Args:
    indicator:  the result of a boolean indicator function
    state:  the current state of the optimization variables

  Returns:
    the state with updated p_cov
  '''
  c_cov = adaptation.c_cov
  discount = 1 - c_cov
  complements = sqrt(c_cov * (2-c_cov) * selection.mu_eff)
  normal = (state.means - state.means_prime) / state.sigma
  state.p_cov = discount * state.p_cov + indicator*complements*normal
  return state


def get_intermediate_pool(pool, state):
  '''Helper function for getting the displacement of the most fit solutions.

  Args:
    pool:  an ndarray of candidate solutions
    state:  the current state of the optimization variables

  Returns:
    the displacement of the most fit solutions
  '''
  return 1 / state.sigma * (pool[:, :selection.mu] - state.means_prime)


def update_C(pool, indicator, state):
  '''Updates the covariance matrix C for the current state.

  Args:
    pool:  an ndarray of candidate solutions
    indicator:  the result of a boolean indicator function
    state:  the current state of the optimization variables

  Returns:
    the state with updated covariance C
  '''
  c_1, c_mu, c_cov = adaptation.c_1, adaptation.c_mu, adaptation.c_cov
  discount = 1 - c_1 - c_mu
  c_loss = (1-indicator) * c_cov * (2-c_cov) * state.C
  rank_1 = c_1 * (state.p_cov @ state.p_cov.T + c_loss)
  rank_min_mu_n = c_mu * pool @ np.diag(selection.weights) @ pool.T
  state.C = discount * state.C + rank_1 + rank_min_mu_n
  return state


def update_sigma(state):
  '''Updates sigma for the current state.

  Args:
    state:  the current state of the optimization variables

  Returns:
    the state with updated sigma
  '''
  c_sigma, damp, chi_N = adaptation.c_sigma, adaptation.damp, adaptation.chi_N
  unbiased = np.exp(
      (c_sigma/damp) * (np.linalg.norm(state.p_sigma) / chi_N - 1))
  state.sigma = state.sigma * unbiased
  return state


def eigen_decompose(evals, state):
  '''Periodically performs eigen decomposition on the covariance matrix.

  Args:
    evals:  the number of solutions evaluated so far
    state:  the current state of the optimization variables

  Returns:
    the state with updated eigenevals count, covariance matrix, B, D, and
    inverse square root of the covariance matrix
  '''
  state.eigenevals = evals
  state.C = np.triu(state.C) + np.triu(state.C, k=1).T
  state.D, state.B = np.linalg.eig(state.C)
  state.D = np.sqrt(state.D)
  state.C_inv_sqrt = state.B @ np.diag(state.D**(-1)) @ state.B.T
  return state


def update_results(pool, fitnesses, results):
  '''Helper function for tracking the best solution and fitness during the
  optimization.

  Args:
    pool:  a sorted ndarray of candidate solutions where solutions are sorted
      by their fitness
    fitnesses:  a sorted ndarray of fitnesses corresponding to the candidate
      solutions
    results:  the current best results of the optimization

  Returns:
    updated best results
  '''
  if fitnesses[0] < results.fitness:
    results.solution = pool[:, 0]
    results.fitness = fitnesses[0]
  return results


def optimize(state):
  '''Main driver of the CMA-ES optimization

  Args:
    state:  the initial state of the optimization variables

  Returns:
    the best results found during the optimization
  '''
  results = Results(None, np.inf)
  evals = 0
  while evals < constants.convergence_evals:
    sigma, means, B, D = state.generation_tuple()
    pool = np.zeros((constants.N, selection.lamda))
    for i in range(selection.lamda):
      pool[:, i] = means.T + sigma * B @ (D * np.random.randn(constants.N))
    fitnesses = get_fitnesses(pool)
    evals += selection.lamda
    pool = pool[:, fitnesses.argsort()]
    fitnesses.sort()
    state = update_means(pool, state)
    state = update_p_sigma(state)
    indicator = get_indicator(evals, state)
    state = update_p_cov(indicator, state)
    intermediate_pool = get_intermediate_pool(pool, state)
    state = update_C(intermediate_pool, indicator, state)
    state = update_sigma(state)
    if evals - state.eigenevals > adaptation.decompose_threshold:
      state = eigen_decompose(evals, state)
    results = update_results(pool, fitnesses, results)
    exploding_scale = D.max() > 1e7 * D.min()
    if fitnesses[0] <= constants.convergence_fitness or exploding_scale:
      break
  return results


def main():
  config = load_config()
  state = init(config)
  results = optimize(state)
  print(results)


if __name__ == '__main__':
  main()
