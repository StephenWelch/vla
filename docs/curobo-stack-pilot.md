# cuRobo stacking pilot

Scope: `cube-double-v0`, task 5. CEM remains available and is the default; other cuRobo task combinations fail explicitly. This compares complete demonstration workflows: cuRobo also changes the controller, so differences cannot be attributed solely to the planning algorithm.

## Run

From the repository root in the OGBench WSL environment:

```bash
OGBENCH_CUROBO=1 bash projects/ogbench-mjwarp/scripts/setup-wsl.sh
export MUJOCO_GL=egl
python -m ogbench_mjwarp.pilot --config configs/ogbench/curobo-stack-pilot.yaml
```

Override `--output` for each experiment. The recipe runs 50 matched reset seeds, batch sizes 1 and 32, and three repeats per backend: 600 attempted episodes. Three separate warmup resets precede each trial. Initialization/kernel capture is timed separately. Trials alternate backend order and run sequentially on one GPU. Diversity perturbations are disabled; upstream waypoint sampling remains seeded. There is no success-quota refill.

For a single state-only collection:

```bash
ogbench-mjwarp generate --env cube-double-v0 --task-ids 5 --planner.backend curobo \
  --episodes 2 --planner.episodes 2 --max-steps 1000 --no-record-images --output outputs/stack/raw
ogbench-mjwarp rerender --source outputs/stack/raw --output outputs/stack/rendered
ogbench-mjwarp export --source outputs/stack/rendered --outcome success \
  --require-contact-valid --output outputs/stack/dataset
```

## Contracts and physical checks

cuRobo is pinned to v0.8.0, commit `4ea77366ca48ee453e7df139e39fa6532af49f3b`, using its batched IK and trajectory optimizer with per-world obstacles. Each phase allows three planning attempts. Robot joint order, URDF transforms, tool frame and conservative collision spheres are derived from the pinned MuJoCo model. Tool position and rotation are checked against MuJoCo before generation.

cuRobo emits six absolute arm angles in radians and normalized gripper closure, bounded by actuator limits and controller velocity/acceleration limits. Archives record the targets actually applied. Raw recordings use `ogbench-rollouts-3`; LeRobot exports use `ogbench-mjwarp-3`. Dataset manifests and checkpoints retain `action_profile`, including names, units, bounds and controller limits. Training, replay and evaluation bind that contract explicitly. CEM retains its five native Cartesian delta actions and v2 format.

Upstream pick/place waypoints define phases. Actual tracking gates phase completion; grasp/release phases dwell, failed lifts and dropped objects invalidate the rollout. The carried cube is attached only to the planner's collision model, using its measured pose; physical simulation never welds it to the hand. Intentional grasp/support segments exempt relevant obstacles in planning; exact MuJoCo contact guards remain active throughout execution. The conservative gripper cover overlaps the neighboring wrist joint, so that one mechanical assembly pair is excluded from self-collision checking.

The pilot appends a one-second hold after a completed goal and audits both backends identically. Acceptance requires native goal success, released gripper, cube-to-cube support contact, upright/aligned cubes, no robot contact with either cube, penetration below 3 mm, and less than 3 mm movement during the final second. Native success and accepted stable demonstrations are reported separately.

## Outputs and interpretation

`results.json` records per-trial success counts, Wilson confidence intervals, paired outcomes, accepted demonstrations/minute, planning time per simulated second, physics/transfer/archive times, latency summaries, target smoothness and Torch GPU allocation peaks. `source.json` fingerprints the implementation; `status.json` records completion and `report.md` summarizes the finished trials. Archive write times sum worker durations and can overlap wall time. Repetitions reuse the reset bank and must not be pooled as 150 independent scenarios. W&B retains metrics and reports.

Rendering follows timed generation. The first ten reset IDs are selected in advance for both backends. Videos show front and wrist views side by side at 640x480 per view. Additional accepted cuRobo episodes from the first batch-32 repeat are rendered for export. Rendering streams compressed arrays instead of holding whole batched videos in memory. No old dataset migration is performed.

Accepted cuRobo episodes feed a two-update ACT integration check and a bounded two-episode simulator evaluation. These check the seven-action interface; they do not establish learned-policy performance.

## Initial validation

The two-reset preflight logged to [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/dy9xbjqx). The matched reset fingerprints agreed. This is a wiring check, not a reliable speed/quality estimate; subsequent retreat handling and memory fixes are not represented in these measurements.

| Workflow | Stable stacks | Generation wall time after initialization | Valid demos/minute |
| --- | ---: | ---: | ---: |
| CEM, batch 2 | 2/2 | 104.66 s | 1.15 |
| cuRobo, batch 2 | 1/2 | 22.17 s | 2.71 |

A successful cuRobo rollout was rendered, inspected and exported at 640x480. ACT completed two updates on that seven-action dataset; native LeRobot evaluation loaded the checkpoint and ran two 20-step rollouts (0/2 task successes, both contact-valid). This validates the interface, not learning quality. Artifacts are under `E:\vla-ogbench\curobo-pilot`, including `smoke-dataset-01`, `act-smoke-02` and `act-eval-smoke-01`.

The batch-32 planning smoke test completed with 11.12 GB peak Torch allocation. The upstream 5000-point interpolation buffer initially exhausted 32 GB; the pilot uses 512 points at the native 20 Hz rate, covering the configured trajectory duration. The compatibility suite passed 137 tests, followed by 30 targeted planner/controller/audit, training and CLI tests. Ruff and lockfile checks passed.

The complete comparison started on October 5, 2026 and is [logging to W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/alpvvyuv). Its final results are pending. Output is `E:\vla-ogbench\curobo-pilot\stack-50`; the persistent process log is `E:\vla-ogbench\curobo-pilot\stack-50.log`. The runner writes the final report, videos, dataset and ACT integration artifacts after the timed trials.

## Executor comparison

The original `waypoint` executor remains the default. Select `--planner.curobo.execution timed` to sample the cuRobo position/velocity curve at 20 Hz instead of advancing each intermediate sample on a joint-error threshold. The curve is uniformly slowed to satisfy discrete velocity/acceleration limits, including the final hold. Actuator bounds and the runtime limiter remain active. Phase metadata records the source timestep and applied time scale; archives also include `annotation/requested_action` to distinguish requested targets from applied controls.

The timed mode advances intermediate samples by elapsed control time; phase completion still checks measured tool position. Grasp/release dwell and all contact/drop/stability guards remain active. Each phase currently ends at rest and planning keeps the original rest-to-rest start-state convention. A smooth correction tapers the command-to-measured joint offset over the path to avoid a sharp first interval. This targets within-phase tracking; it does not yet blend independently collision-checked phase paths.

```bash
export VLA_WANDB_NETRC_PATH=/mnt/c/Users/steph/.netrc
python -m ogbench_mjwarp.execution_ablation --config configs/ogbench/curobo-execution-ablation.yaml
```

The recipe compares four reset seeds against the archived batch-1 waypoint trial. Omit the `baseline` setting to regenerate both modes. Choose a fresh output directory for each run. `results.json` records native/contact/stable outcomes, simulated duration, commanded and actual acceleration peaks/RMS, tracking error and substantial velocity reversals. Reversals include intended motion changes and require trace inspection; they are not a standalone jitter score. `*-traces/episode-*-joints.npz` contains aligned time, command, position and velocity arrays. Initial-state fingerprints must match. Tests overlapping the original GPU benchmark assess motion quality only; generation throughput needs an isolated run.

The first aggressive timed variant supplied measured initial velocity and removed clearance/transit endpoint gates. It produced 0/4 accepted stacks versus 3/4 for the archived waypoint baseline ([W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/401zob37), output `E:\vla-ogbench\curobo-pilot\execution-4-02`). Three failed simulator validity checks with finite recorded states; one failed a contact check. Its shorter failed episodes do not establish a speed improvement. Those transition changes were reverted for the conservative continuous-tracking comparison.

The conservative comparison completed on the same four reset seeds ([W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/mnhkskl1), output `E:\vla-ogbench\curobo-pilot\execution-4-03`). Both modes accepted 3/4 stable stacks, on different subsets of seeds. Timed tracking completed all four native goals; one failed the final alignment audit. Mean actual joint acceleration RMS decreased from 0.561 to 0.260 rad/s² and substantial reversals from 1.822 to 0.964 per second. Mean attempted-episode duration was 27.59 versus 27.30 simulated seconds. These four seeds support a smoothness improvement, not a reliable success-rate or throughput estimate. Phase blending and automatic adoption remain deferred. Nine focused CPU tests and Ruff checks passed. W&B includes the episode-zero command/measurement trace plot and per-episode audit tables.
