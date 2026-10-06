## OCBench workflow

Native GPU demonstrations for `block-double-task2-v0` (stack anywhere), deferred batched cameras, LeRobot export and ACT training/evaluation. Native termination and randomization are preserved. Physical audits filter exports; released stable-stack status is a separate diagnostic.

```bash
bash projects/ocbench-mjwarp/scripts/setup-wsl.sh
source ~/.venvs/vla-ocbench/bin/activate
export MUJOCO_GL=egl
vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml
```

The recipe retains 500 attempts and trains ACT on physically audited native successes. Native failures passing the same audits are exported separately. Every fifth attempt is assigned to validation before filtering. W&B logs to `vla-ocbench`; use `VLA_WANDB_NETRC_PATH` for an existing Windows netrc from WSL.

```bash
ocbench-mjwarp generate --output outputs/ocbench --episodes 32
ocbench-mjwarp render --source outputs/ocbench
ocbench-mjwarp export --source outputs/ocbench --output outputs/ocbench/dataset
MUJOCO_GL=glfw ocbench-mjwarp view --root outputs/ocbench --episode 0
```

Actions are normalized joint/gripper deltas at 50 Hz. OGBench absolute-target checkpoints are incompatible. Front/wrist images are 640×480; the policy state contains only proprioception. Raw archives retain N+1 simulator states, native outcomes, substep robot/environment penetration peaks, sampled plans and retry history. These numerical checks do not certify visual mesh or continuous-time collision freedom.

Generation and completed render batches are restartable with the same configuration. Partial exports require a fresh output. Datasets, checkpoints and full videos remain outside Git. See the [planner comparison](../../docs/ocbench-planner-comparison.md).
