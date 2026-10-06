# Spline stacking demonstrations

The opt-in `spline` backend generates unseeded attempts for `cube-double-v0`, task 5. CEM and the original cuRobo backend remain available. It uses the existing seven-dimensional joint-target/gripper action profile, MJWarp physics, batched renderer, and LeRobot integration. No human trajectory, object welding, recovery policy, or perception-error model is involved.

## Generate, render, and export

From the repository root in WSL, using the OGBench environment with cuRobo installed:

```bash
export PATH=/mnt/e/vla-ogbench/venv/bin:$PATH
export MUJOCO_GL=egl
python -m ogbench_mjwarp.cli generate --config configs/ogbench/spline-stack.yaml
python -m ogbench_mjwarp.cli rerender \
  --source /mnt/e/vla-ogbench/spline-stack/raw \
  --output /mnt/e/vla-ogbench/spline-stack/rendered --batch-size 32
python -m ogbench_mjwarp.cli export \
  --source /mnt/e/vla-ogbench/spline-stack/rendered \
  --output /mnt/e/vla-ogbench/spline-stack/successes --quality validated-success
python -m ogbench_mjwarp.cli export \
  --source /mnt/e/vla-ogbench/spline-stack/rendered \
  --output /mnt/e/vla-ogbench/spline-stack/failures --quality valid-failure
```

Defaults are 100 attempts, 32 execution worlds, and an 80% moderate / 20% challenging sampler. Counts are attempts, without success-quota refill. CLI arguments override YAML. For example, `--episodes 4 --planner.episodes 4 --planner.spline.variation nominal` runs a small deterministic-parameter check. Choose a new output when changing configuration or source version; completed episodes are restartable within the same run configuration.

For interactive playback from PowerShell:

```powershell
.\scripts\view-ogbench.ps1 -Root E:\vla-ogbench\spline-stack\raw -Episode 0
```

The ordinary viewer also opens failed attempts. LeRobot export requires rendered images; raw simulator states alone are sufficient for the viewer.

## Motion and failure semantics

The stacking adapter supplies object-relative poses, contact expectations, gripper commands, and stops. The shared planner samples grasp candidates, checks IK and exact grasp geometry, and initializes paths with cuRobo. It blends the paths between contact stops into quintic joint splines. Transit event boundaries do not require stopping or waiting for measured tracking error. Closure and release have scheduled smooth gripper commands and dwell. The entire attempt is planned before physical execution.

The initial budget is four grasp candidates, 16 IK seeds, two trajectory seeds, and one cuRobo planning attempt per path. Feasible grasp candidates are ranked by squared joint travel as a cheap path-length proxy. Nominal mode searches deterministic cube symmetries; randomized modes sample symmetry, yaw, and tilt. Candidate screening and selection bias the realized grasp distribution; every proposal and the selected indices are recorded.

| Parameter | Moderate | Challenging |
| --- | --- | --- |
| Grasp XY offset | ±5 mm | ±8 mm |
| Yaw around a cube symmetry | ±10° | ±15° |
| Tilt cone, uniform solid angle | 10° | 20° |
| Placement XY offset | ±2 mm | ±8 mm |
| Free-space waypoint offset | ±10 mm | ±30 mm |
| Requested segment duration multiplier | 0.85–1.20 | 0.65–1.40 |

Each object has independent placement, path, and segment timing samples. Grasp proposals and the sampler stratum are shared within an attempt; each transfer selects its feasible proposal. Contact dwell is uniform from 0.2 to 0.6 seconds. Full hand-to-object transforms determine placement wrist poses, preserving the intended object orientation under tilted grasps. A tilt-dependent pinch-height offset reduces pad/table interference; exact geometry screening remains authoritative.

Curves are retimed for 1 rad/s velocity, 2 rad/s² acceleration, and 50 rad/s³ jerk, with a final discrete check across the whole command sequence. Fitted paths are checked against joint limits, event-pose tolerances, and MuJoCo collision geometry at dense intermediate samples. Sampling is not a continuous collision proof. The planning check uses conservative open-jaw geometry; execution retains the existing exact contact/penetration guards at physics substeps. Requested duration scales may be lengthened by retiming.

Execution never replans after an ordinary missed grasp, drop, or misplaced block. It finishes the scheduled motion and one-second hold. Numerical invalidity, capacity overflow, and prohibited penetration stop execution. Finite objects leaving the recording bounds are labeled `out_of_bounds`, separately from numerical failure. Grasp/drop event detectors use observed lift and tool/object separation; the metadata records their thresholds. They are diagnostic labels, not contact-sensor ground truth.

The final stable-stack audit requires a released, upright, aligned, supported stack throughout the last second. Native goal completion is retained separately. Smooth physically valid failures can be useful, but no metric alone establishes that a failure looks humanlike.

## Data contract

### Optional waypoint relaxation

Add `--planner.spline.waypoint-relaxation 0.1` to generation to smooth within a 0.1-radian joint-space corridor around the original curve. The default is zero, preserving the original interpolating fit. This changes the planned joint targets; it does not loosen the execution controller or its contact guards.

The fitter minimizes integrated squared jerk over the spline knots. Event `anchor` flags keep required poses exact, independently of contact expectations and whether an event requests a stop. The stacking adapter anchors grasp, closure, placement, release, and final retreat; intermediate lift and transit poses are guides. The same fitting and validation logic works for any adapter supplying these constraints, without phase-name branches in the smoother.

The B-spline coefficient bound limits deviation along the entire curve, rather than only at the knots. Guides must also remain within `guide_position_tolerance` (default 0.03 m) and `guide_rotation_tolerance` (default 0.25 rad) of their requested poses. Anchors retain the existing 3 mm / 0.04 rad checks, and joint, derivative, carried-object, and collision checks remain active. The bounded corridor itself does not establish clearance: every proposal is geometrically validated. Collision validity remains subject to the dense sampling resolution and execution tracking.

The planner tries the requested relaxation, half that value, and finally the original curve. Each fallback is checked; if none passes, the attempt is rejected. Per-group `smoothing` metadata records every tried bound, rejection reason, accepted joint-deviation bound, event-pose errors, and nominal jerk integral before/after optimization. Failed task attempts remain governed by the ordinary outcome filters.

An eight-reset comparison (seeds 62000–62007, eight worlds, mixed sampler) tested zero versus 0.1 rad relaxation: [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/ij9uw7p0). Both produced seven validated successes and the same one planning rejection, with no execution-invalid attempts. Across completed episodes, mean measured joint jerk RMS fell from 1.445 to 1.049 rad/s³, acceleration RMS from 0.234 to 0.194 rad/s², and pause fraction from 17.2% to 15.6%. Mean duration changed from 27.64 to 27.31 seconds. Of 70 accepted fitting groups, 42 used 0.1 rad, 26 used 0.05 rad, and two fell back to zero; rejected smoothing proposals exceeded event-position tolerances.

This is a small motion-quality check, not evidence of a reliability or throughput improvement. Tests overlapped baseline initialization, so wall timings are not an isolated performance comparison. Relaxation remains opt-in. Artifacts and the exact driver are under `E:\vla-ogbench\spline-pilot\smoothing-8`; paired videos show episodes 1 and 2. The focused spline, cuRobo, and CLI suite passed 30 tests, including whole-curve bounds, fixed anchors, and collision-rejection/backoff behavior; Ruff passed.

### Local retiming and event speeds

Add `--planner.spline.retiming local` to enable faster timing and blend the retreat between transfers into the next approach. It works with `--planner.spline.waypoint-relaxation 0.1`; both features remain opt-in. Grasp, closure, placement, release, and the final episode endpoint still stop. Closure/release dwells are not shortened, and retreat remains a geometric anchor even when it no longer requests a stop.

```bash
python -m ogbench_mjwarp.cli generate --config configs/ogbench/spline-stack.yaml \
  --output /mnt/e/vla-ogbench/spline-stack-local/raw \
  --planner.spline.waypoint-relaxation 0.1 --planner.spline.retiming local \
  --planner.spline.max-jerk 15
```

The shared retimer changes the clock, not the geometric curve. It compares a compressed uniform clock with a smooth locally varying clock seeded from joint derivative demands. Chain-rule derivative checks, event speed constraints, and discrete 20 Hz command checks determine any additional slowdown. The shorter valid clock wins. This is a bounded heuristic, not a globally time-optimal solver; it does not optimize torque or contact forces. Existing collision validation includes the retimed command samples, and runtime contact guards and final audits remain active.

Events can carry independent `max_linear_speed` (m/s) and `max_angular_speed` (rad/s) constraints at their arrival poses. The stacking adapter uses `transit_linear_speed: 0.3` and `transit_angular_speed: 1.5` in local mode. MuJoCo site Jacobians translate these constraints into clock-speed bounds. These are limits at event arrivals, not Cartesian limits along the entire segment; joint limits apply throughout. Zero-speed requirements use `stop`. The retimer accepts a task-independent speed-constraint callback and contains no reach/grasp/drawer branches. The current simulator adapter remains stacking-only; other task adapters must provide their event programs and scene/contact semantics.

Episode metadata records the requested mode, selected clock, candidate durations, event speed/limit ratios, realized arrivals, and each event's `segment_time_scale`. The legacy `time_scale` now denotes the group's total duration ratio; it is not a local speed multiplier. Requested random duration scales are retained, but local retiming can change their realized ratios. The seven-dimensional commands and replay states retain the complete executed trajectory.

An eight-reset comparison used seeds 62000–62007, eight worlds, and 0.1 rad waypoint relaxation in both modes: [W&B and paired videos](https://wandb.ai/rlgoats/vla-ogbench/runs/nczwfabg). Both produced seven validated successes and the same planning rejection, with no invalid execution attempts. Local mode reduced mean completed duration from 27.31 to 21.09 seconds and measured paused time from 4.26 to 3.49 seconds. It increased measured joint jerk RMS from 1.05 to 1.97 rad/s³ and acceleration RMS from 0.194 to 0.323 rad/s². Pause fraction increased from 15.6% to 16.7% because total duration fell more than paused time. These results establish a speed/smoothness tradeoff, not an unqualified smoothness improvement.

Artifacts and the exact comparison driver are under `E:\vla-ogbench\spline-pilot\retiming-8-v2`. Local clocks won for 20 fitting groups; compressed uniform clocks won for 43 (including dwells). The fused between-transfer retreat removes one fitting group per completed episode. The shared spline/cuRobo/CLI suite passed 32 tests, with additional local-clock collision-backoff and CLI-override checks passing afterward. The retimer was also tested on an uneven synthetic path with an independent task-space speed constraint; this is not a drawer simulation test.

Lowering only `max_jerk` from 50 to 15 rad/s³ produced 21.37-second episodes with 1.86 rad/s³ measured jerk RMS, retaining seven successes and one planning rejection: [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/eu284rcr), artifacts `retiming-jerk15-8`. The small change shows that limiting peak commanded jerk alone does not recover the slower baseline's measured smoothness. These tuning runs reuse the same eight seeds and are not additional independent reliability observations.

A final gentler profile added `--planner.curobo.max-acceleration 1.0` to local retiming and jerk 15: [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/85ztuw9v), artifacts `retiming-balanced-8`. It again produced seven successes and one planning rejection, but mean duration rose to 27.55 seconds, with measured jerk RMS 1.07 rad/s³ and paused time 4.21 seconds. Commanded jerk RMS fell from the baseline's 1.30 to 1.13 rad/s³. This profile roughly recovers the baseline's speed and measured smoothness rather than improving both. The acceleration budget also affects cuRobo initialization, so this last trial compares configurations, not timing alone. All four settings had zero invalid execution attempts; wider-task reliability and perceived humanlikeness remain unestablished.

### Frame and episode annotations

Every raw attempt remains available, including planning rejections (which may contain only a hold frame), invalid simulations, and timeouts. `--quality validated-success` selects completed, physically valid stable stacks. `--quality valid-failure` selects completed, physically valid attempts that fail the stable audit. Neither includes rejected plans or truncated attempts. Legacy rows without quality evidence match neither quality filter. Existing outcome/contact filters still work.

Annotation schema 2 adds `annotation.target_pose` (world xyz and quaternion wxyz) and `annotation.task_error` to the LeRobot frame columns. The existing five-value reference remains for compatibility; use the full pose for tilted targets. Archives retain requested actions, applied actions, and N+1 simulator states for N action frames.

Episode metadata preserves sampling configuration, independent random streams, sampler stratum, every grasp proposal, selected candidates, object-relative transforms, measured grasp transforms, requested timing, realized event arrivals, observed event frames, native success, stable-stack audit, physical validity, completion, and motion metrics. The cuRobo optimizer uses the run seed and may vary with batching; reproducible factor samples do not imply bitwise-identical GPU trajectories.

## Diverse collections

`configs/ogbench/spline-stack-diverse.yaml` collects 500 stacking attempts with a 60% broad / 40% challenging mix. It adds uniform-area XY approach disks (4/8 cm radius), independent pick/preplace heights (10–20/8–23 cm), larger transport offsets (±4/8 cm), grasp XY offsets (±8/14 mm per axis), full yaw coverage through cube symmetries and ±45° yaw, and uniform-solid-angle tilt cones (20/35°). Placement variation remains at the existing ±2/8 mm bounds.

Six grasp proposals are screened using the existing geometry/IK checks; independent random priorities choose among feasible proposals, replacing the shortest-joint-motion preference for this profile. Feasibility and downstream outcome filtering still bias the realized distribution. All proposals, priorities, selected indices, requested distributions, and random-stream seeds remain in the episode metadata.

Each attempt samples an execution-speed multiplier uniformly from 0.65–1.0 (broad) or 0.5–1.0 (challenging). It slows the already constrained clock, survives local retiming, and cannot raise motion limits. Gripper dwells remain independently sampled at 0.2–0.65 seconds. Random segment-duration ratios are also retained; realized arrivals and timing factors record what the retimer actually executed. `approach_vectors` records world-frame offsets from each pick/place pose; the fields are additive to annotation schema 2.

```bash
export VLA_WANDB_NETRC_PATH=/mnt/c/Users/steph/.netrc
python -m ogbench_mjwarp.collect --config configs/ogbench/spline-stack-diverse.yaml
```

This Tyro CLI accepts YAML defaults and explicit overrides. The default output is `/mnt/c/Users/steph/data/vla/ogbench-stack-diverse-500`; override `--output` or `OGBENCH_COLLECTION_ROOT`. Generation runs in a separate process to release planner memory before rendering. The collector retains all raw attempts, renders only audited completed successes/failures at 640×480 per camera, exports separate LeRobot datasets, verifies they load, and uploads sampled videos to W&B. Every fifth reset seed is assigned to validation before filtering, consistently across both outcome datasets. A fresh configuration requires a fresh output; completed render batches are reusable, while incomplete batches/exports fail explicitly rather than being mistaken for complete data.

`collection.json`, `generate.json`, `factor-summary.json`, `dataset-split.json`, `exports.json`, and `status.json` document the collection. Factor coverage is reported for all attempts, successes, and valid failures separately. A requested count means attempts, not a quota of successes; rejected/truncated/invalid attempts never enter either filtered dataset. Camera rendering and dataset export remain separate from rollout generation.

The separate 16-attempt pilot (seeds 71000–71015) produced three validated successes, eleven physically valid failures, one planning rejection, and one contact-invalid attempt: [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/a1iqv1nb). Selected grasp tilts reached 34.6°, approach offsets reached 7.9 cm, and sampled execution speeds ranged from 0.579 to 0.974. Successes covered a narrower selected tilt range, 8.4–15.8°. This aggressive profile deliberately retains a large failure population; neither the nominal ranges nor this small pilot establish broad success coverage. Pilot outputs live under `C:\Users\steph\data\vla\ogbench-stack-diverse-pilot-16` and are excluded from the 500-attempt collection. The focused non-dataset suite passed 15 tests, including persistent speed scaling, unchanged dwell timing, approach construction, and outcome-filtered coverage reports; Ruff passed.

## Matched pilot

```bash
export VLA_WANDB_NETRC_PATH=/mnt/c/Users/steph/.netrc
python -m ogbench_mjwarp.spline_pilot --config configs/ogbench/spline-stack-pilot.yaml
```

The driver runs each trial in a fresh process: a single-world smoke check, a 32-world memory check, ten tuning resets, and fifty separate matched resets repeated twice. It compares timed cuRobo, nominal splines, moderate variation, and the mixed sampler. It stops if the nominal smoke cannot finish an attempt or a trial exceeds the 28 GiB sampled device/Torch memory budget. Device samples include other GPU users and are not a guaranteed allocation peak; run the benchmark alone. `--wait-pid PID` queues measured work behind an existing Linux process.

W&B and local `trial.json`/`results.json` files report validated successes/minute, valid failures/minute, rejected plans, invalid attempts, per-stratum counts, duration, pause fraction, acceleration/jerk, memory, and timing breakdowns. Audits and archive writing count toward generation time; warmup and rendering are separate. Repeats share scenarios and are not additional independent observations.

After measured trials, mixed-sampler results are batch-rendered and exported to separate `datasets/successes` and `datasets/failures` directories. Empty result classes are reported explicitly. Review videos include up to ten successes and ten valid failures per mixed trial. A reset-seed split is assigned across both datasets before export; training's scene splitter honors those labels and rejects conflicting or incomplete labels. The files `dataset-split.json`, `exports.json`, `report.md`, and `status.json` summarize the run. No policy training is started by this pilot.

## Initial verification

Functional checks ran alongside an older GPU benchmark, so their timings do not establish a speedup:

- Nominal single-world check: one stable success.
- Default mixed sampler, four worlds: four stable successes, including a selected 16.5° tilted grasp.
- A five-attempt refill check repeated those four seeds and added one more: the first four again succeeded; the additional attempt left the finite recording bounds and was excluded from demonstration exports.
- Stress check with placement offsets expanded to ±25 mm: one stable success, two physically valid completed failures, and one planning rejection. This stress setting is not the generation default.
- A 32-world planning/execution smoke check fit 29 complete programs and rejected three candidates. It deliberately stopped after two control steps, so it is not an outcome-rate trial. Torch reserved memory peaked at 5.70 GiB; sampled device usage was 7.60 GiB, below the 28 GiB budget.

Artifacts are under `E:\vla-ogbench\spline-pilot`. The stress attempts are in `smoke-errors-01`; completed failure episodes are 0 and 2. Test coverage includes deterministic factor sampling, full-pose grasp/placement transforms, spline bounds and blending, missed-grasp/drop continuation, quality filters, annotation/export round trips, and grouped splits. Full matched throughput conclusions remain pending the isolated pilot.

The combined spline, cuRobo, CLI, dataset, randomization, and training-hook suite passed 56 tests. Ruff checks passed. Existing simulator precision, gimbal-lock, and collision-shape warnings remain.

The stress check was rendered at 640×480 per camera and exported to `smoke-successes` (one episode, 716 frames) and `smoke-failures` (two episodes, 1,182 frames). Both load seven-dimensional action chunks and full-pose annotations through LeRobot. Paired front/wrist review videos are in `videos/smoke-000000.mp4` through `smoke-000002.mp4`; sampled frames show the failed attempts ending with offset/tipped blocks after the arm retreats.

The queued matched comparison ([W&B run](https://wandb.ai/rlgoats/vla-ogbench/runs/jqpvh331)) and the old CEM/cuRobo benchmark were stopped at the user's request. They are not running or queued. An interrupted measured trial requires a fresh output so partial wall time cannot inflate throughput; completed trials are reusable.

## Isolated 32-attempt test

The default mixed sampler was tested alone on the GPU on reset seeds 62000–62031, with 32 concurrent worlds and a 1,600-step limit: [W&B](https://wandb.ai/rlgoats/vla-ogbench/runs/0v41p1la). The output is `E:\vla-ogbench\spline-pilot\test-32`; `run.py` records the exact invocation and postprocessing.

| Outcome | Attempts |
| --- | ---: |
| Validated stable success | 20 |
| Physically valid completed failure | 10 |
| Planning rejection | 1 |
| Out-of-bounds truncation | 1 |
| Contact-limit or numerical failure | 0 |

Generation took 135.36 seconds after 34.32 seconds of initialization: **8.87 validated successes/minute**, plus **4.43 valid failures/minute**. These timings include validation and archive writing, and exclude subsequent rendering/export. Sampled device memory peaked at 7.60 GiB. Planning accounted for 99.30 seconds, including 92.78 seconds before the first execution step; physics took 12.26 seconds.

Nine completed failures missed only the strict 12 mm stack-alignment check. One also failed support, uprightness, and native-goal checks. Moderate samples produced 15 successes and nine valid failures across 24 attempts; the challenging tail produced five successes, one valid failure, and the two excluded attempts across eight attempts. This small test does not establish a difference between strata or a speedup over another planner.

Raw states for all attempts are retained under `raw`. The review/export subset contains successful episodes 1 and 2 and failed episodes 8 and 12; its videos and separate LeRobot datasets are under `videos`, `successes`, and `failures`. These small exports are integration examples, not a training/validation dataset.
