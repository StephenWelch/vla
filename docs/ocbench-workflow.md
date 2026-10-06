# OCBench collection and ACT workflow

The native `block-double-task2-v0` controller replaces the planner for the new stacking collection. Its 50 Hz actions, randomization, retries and native stopping conditions are preserved. Existing OGBench planners and checkpoints remain available separately. OCBench is pinned to `e2cd2f72110b66bd65afab1b855d81ebc73aeacc`; dependencies are locked in the new simulator project.

## Data contract

Each attempt retains N actions and N+1 states, including the terminal state before upstream parking. Metadata includes reset/oracle seeds, train/validation assignment, native success/health, completion reason, contact peaks, numerical/capacity validity, stable-stack diagnostics, native timing distributions and sampled keyframe plans. Plans include world poses, gripper targets, tangents, mistake flags, RNG counters and retry counts. These are planned motion annotations, not measurements of actual grasp contact points.

Exported successes require native success, native workspace health, finite physics, no capacity overflow and robot/environment penetration within 1 mm for non-pad contacts and 3 mm overall. Valid native failures use the same checks. Solver/line-search iteration-budget flags are retained separately and do not alone invalidate an attempt. The contact audit covers simulator collision geometry at physics substeps; it does not certify visual mesh, self-collision or continuous-time collision freedom. Stable released stacks are a separate diagnostic and are not required: native episodes can terminate while the gripper still holds a block.

Actions are seven normalized joint/gripper deltas with native scales, not the existing OGBench absolute targets. Dataset/checkpoint camera and action profiles must match. Policy inputs contain 18 proprioceptive values and two RGB images; privileged object state stays in replay/annotations. Images are 640x480, with native OCBench front and wrist optics. Rendering replays stored states in GPU batches, updates ray-tracing bounds, and writes bounded-memory videos before LeRobot export.

## Verification

The 32-attempt integration pilot (reset seeds 82000-82031) produced 26 native successes, 23 contact-audited successes, three valid failures and six contact-invalid attempts. This pilot preceded addition of explicit capacity-overflow recording; production records that evidence as well. See [collection W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/i989lbt9).

- All 32 archives passed N+1-state and non-parked terminal-state checks.
- Eleven usable attempts were rendered; front/wrist samples were visually inspected at initial, intermediate and final states.
- Four successes (7,536 frames) and one failure (2,500 frames) exported and loaded through LeRobot. Successes split into three train and one validation episode.
- A two-update ACT smoke run completed separate train/validation loss probes and one three-step rollout per split. Checkpoint reload and standalone evaluation also passed. [Training W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/cr9me5mo). This verifies integration, not learned task competence.
- Shared training/contact/config/tracking regressions passed 49 tests; OCBench contract and native/instrumented GPU parity passed four. Rendering/CLI tests passed 18 initially; two resume tests exposed tuple/list config serialization and passed after canonicalizing the saved-run comparison. Ruff and lockfile checks passed.

Pilot artifacts are under `C:\Users\steph\data\vla\ocbench-integration-pilot`. The compatible existing WSL environment `/mnt/e/vla-ogbench/venv` was used for these runs to avoid duplicating large GPU dependencies on the nearly full E: drive. The new setup script creates a separate environment for a fresh installation. PyAV is used for video writing and decoding; an optional TorchCodec shared-library warning does not prevent this path.

## Production recipe

```bash
MUJOCO_GL=egl vla-ocbench-pipeline --config configs/ocbench/stack-act.yaml
```

The recipe uses 500 attempts, 32 worlds, reset seeds 83000-83499 and oracle seeds 93000-93499. Every fifth attempt is validation, assigned before outcome filtering. All raw attempts remain; only audited native successes/failures are rendered/exported. The pipeline estimates storage before rendering, checks a 4 GiB reserve between batches, and verifies exported datasets load.

ACT trains from scratch for 20,000 updates on audited native successes, with batch size eight and 40-step action chunks. Normalization statistics use training episodes only. Loss probes run every 1,000 updates; policy rollouts run every 2,000 on ten fixed train and ten fixed validation reset states, up to 2,500 steps each. Native success, audited success, contact/physics validity, truncation and stable-stack diagnostics are separate metrics. Evaluations use the policy without oracle recovery. Checkpoint retention keeps latest and best validation checkpoints.

The full pipeline was launched under a hidden Windows WSL host process: [collection W&B](https://wandb.ai/rlgoats/vla-ocbench/runs/o3voj764). Its first 32 attempts produced 28 native successes, 23 audited successes, three valid failures and six invalid attempts. Output is `C:\Users\steph\data\vla\ocbench\stack-500`; `status.json`, `pipeline.log`, `pipeline.stderr.log` and `tracking.json` report progress. ACT starts automatically after rendering/export. A launch is not a completed collection or training result.

Generation resumes committed attempts with the original batch layout; completed camera videos are reused. Incomplete exports require a fresh output. Interrupted training resumes through `vla-ocbench-train --config <run>/train.json --resume <checkpoint>/pretrained_model`; the pipeline does not silently restart or overwrite a training run.
