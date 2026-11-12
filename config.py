"""
Central configuration for MEP_evolve_VRP.

Defaults follow the paper's reported evolution budget where this repository
can represent it: 10 generations, 10 offspring per generation, and survivor
population size 5. The evaluator in this repository is still a custom GA
prototype, NOT the paper's original HGS evaluator.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Gemini: set GEMINI_API_KEY in the environment; never commit a real key.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Evolution budget (paper-aligned values for T, offspring, and survivor N).
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "10"))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "5"))
OFFSPRING_PER_GENERATION = int(os.getenv("OFFSPRING_PER_GENERATION", "10"))

# Benchmark/evaluator. Despite the legacy name HGS_TIME_LIMIT, the current
# fitness.py runs a custom GA + PyVRP LocalSearch, not original HGS.
DATA_DIR = str(ROOT_DIR / "data")
VRP_INSTANCES = [
    "X-n101-k25.vrp",
]
HGS_TIME_LIMIT = float(os.getenv("EVALUATION_TIME_LIMIT", "10"))
NUM_EVAL_RUNS = int(os.getenv("NUM_EVAL_RUNS", "5"))
BASE_RANDOM_SEED = int(os.getenv("BASE_RANDOM_SEED", "42"))

GENERATED_CODE_DIR = str(ROOT_DIR / "generated_code")
RESULTS_DIR = str(ROOT_DIR / "results")

# Basic validation catches configuration mistakes before an expensive run.
if NUM_GENERATIONS < 1:
    raise ValueError("NUM_GENERATIONS must be >= 1")
if POPULATION_SIZE < 2:
    raise ValueError("POPULATION_SIZE must be >= 2")
if OFFSPRING_PER_GENERATION < 1:
    raise ValueError("OFFSPRING_PER_GENERATION must be >= 1")
if HGS_TIME_LIMIT <= 0:
    raise ValueError("EVALUATION_TIME_LIMIT must be > 0")
if NUM_EVAL_RUNS < 2:
    raise ValueError("NUM_EVAL_RUNS must be >= 2 for a meaningful variability estimate")
