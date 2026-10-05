# SO-101 System Identification: Two-Stage Optimization Plan

## Context

The previous implementation (`scripts/run-so101-sysid.py`) uses a single-stage optimization approach with either `scipy` or `mujoco` optimizer. Testing showed:

- `scipy` optimizer works but is slow (takes ~1-2 minutes per iteration due to sequential residual calls)
- `mujoco` optimizer hangs during execution
- Parameters: 55 total (24 inertials + 18 joint + 12 actuator + 1 delay)

## Two-Stage Approach

The request is to implement a two-stage optimization:

1. **Stage 1: Regression** - Run multiple random initializations to find diverse solutions
2. **Stage 2: CEM Fine-tuning** - Use Cross-Entropy Method to refine the best solutions

### Stage 1: Regression

- Generate N random initial parameter samples (e.g., N=20)
- Run a short optimization from each starting point
- Collect the final parameters and costs
- Select top K candidates (e.g., K=5) with lowest costs

### Stage 2: CEM Fine-tuning

CEM is an iterative stochastic optimization method:
1. Sample M parameter vectors from a Gaussian (mean, covariance)
2. Evaluate costs for all samples
3. Select top P% (e.g., 20%) with lowest costs
4. Update mean and covariance from selected samples
5. Add small "exploration" noise to avoid premature convergence
6. Repeat for G generations

From mjbatch's `examples/rizon_inertia.py`, we can see how to implement CEM for parameter optimization.

## Key Differences from Current Implementation

### Current approach:
- Single optimization from one initialization
- Uses mu_jacobian_fd (parallel finite difference) which can crash in Docker
- Parameters are updated via scipy.least_squares or mujoco.minimize

### Proposed approach:
- Multiple random initializations (regression)
- Sequential residual evaluation (no parallel FD)
- CEM samples are evaluated by running the residual function directly
- No jacobian computation needed (CEM is derivative-free)

## Implementation Plan

### Step 1: Add CEM functions to runner

Create helper functions:
- `sample_parameters(params_dict, n_samples)` - generate random samples around nominal
- `evaluate_cost(x, params, residual_fn)` - call residual and compute sum of squares
- `cem_optimize(residual_fn, nominal_params, n_samples=200, n_elite=20, n_iters=20)` - run CEM

### Step 2: Add regression stage

- `run_regression(params, residual_fn, n_init=20, max_iters=5)` - run short LM from multiple random starts
- Return list of (cost, params) tuples

### Step 3: Integrate into main()

Add command-line flags:
- `--two-stage` - enable two-stage mode
- `--regression-iters` - iterations for initial LM
- `--reg-samples` - number of regression samples
- `--cem-iters` - CEM generations
- `--cem-samples` - samples per CEM iteration
- `--cem-elite` - elite fraction

### Step 4: Output handling

- Save all intermediate results
- Report best final parameters
- Compare regression vs CEM results

## Key Considerations

1. **Computational cost**: CEM can be expensive with many samples. Each sample requires a full residual evaluation.

2. **Parameter scaling**: Some parameters may have very different scales. Need proper normalization in CEM.

3. **Convergence**: CEM can get stuck in local minima. The exploration term (`sigma *= 1.1` or similar) helps.

4. **Docker compatibility**: All optimization is now iterative sequential residual evaluation, avoiding parallel FD crashes.

## Files to Modify

1. `scripts/run-so101-sysid.py` - Add CEM functions and integrate into main
2. `README.md` - Document the two-stage approach

## Reuse Existing Code

- `residual_fn` - already built via build_residual_fn
- `make_sequential_rollout()` - sequential mjbatch rollout (no memory issues)
- `make_modify_residual()` - handles delay and velocity weighting
- `Parameter.as_vector()` / `ParameterDict.update_from_vector()` - for parameter handling