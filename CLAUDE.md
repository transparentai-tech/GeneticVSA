# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Hyperdimensional Evolutionary Algorithms using Vector Symbolic Architectures (VSA) for high-dimensional optimization. The core innovation is implementing genetic algorithms that operate natively in hyperdimensional space (10,000+ dimensions) using VSA operations.

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Quick test of VSA-GA on benchmark functions
python quick_test.py

# Run scaling experiments (dimensions: 10 to 10,000)
python experiments/scaling.py

# Run workshop paper experiments
python experiments/workshop.py

# Run full experimental suite
python experiments/full_suite.py
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_vsa_ga.py

# Run with coverage
pytest --cov=src tests/
```

## Architecture

### Core Components

1. **VSA-GA Algorithm** (`src/vsa_ga.py`)
   - Implements genetic operations in hyperdimensional space
   - Key methods: `encode_solution()`, `decode_solution()`, `vsa_crossover()`, `vsa_mutate()`
   - Uses 10,000-dimensional binary hypervectors by default

2. **Benchmark Functions** (`src/benchmarks.py`)
   - High-dimensional test functions: Rastrigin, Ackley, Schwefel, Sphere
   - Scalable from 10 to 100,000 dimensions

3. **Experiments** (`experiments/`)
   - Dimensional scaling analysis
   - Comparison with CMA-ES and standard GA
   - Transfer learning demonstrations

### Key Implementation Details

- **Hypervector Encoding**: Solutions are encoded as binary hypervectors (±1) of dimension 10,000+
- **Crossover**: Implemented via weighted superposition (bundling) of parent hypervectors
- **Mutation**: Sparse bit flipping in hyperdimensional space
- **Population Size**: Default 50 individuals for high-dimensional problems

## Research Goals

The project aims to demonstrate:
1. GA scaling to 10,000+ dimensions (where CMA-ES fails)
2. Linear or sub-linear scaling with dimension
3. Compositional problem-solving capabilities
4. Transfer learning between optimization problems

## Implementation Priorities

When implementing new features:
1. Maintain simplicity - core VSA-GA should be <200 lines
2. Focus on high-dimensional scaling demonstrations
3. Compare against CMA-ES (fails >1000-D) and standard GA
4. Use established benchmarks (Rastrigin, Ackley, Schwefel)

## Dependencies

Core requirements:
- `numpy` - Vector operations
- `matplotlib` - Plotting results
- `pycma` - CMA-ES baseline comparison
- `pytest` - Testing framework