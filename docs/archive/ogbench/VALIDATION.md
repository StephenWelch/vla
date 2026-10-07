> Archived validation report for the removed OGBench implementation. Commands below are historical.

> Archived v1 results. The datasets and OpenGL recording path are retired; raw recordings and historical checkpoints are retained. Use the v2 training workflow for new experiments.

## Local validation

Checked on 2026-09-26 using an RTX 5090, NVIDIA driver 616.56, and Linux through Docker Desktop/WSL 2. [Machine-readable results](validation/rtx5090.json) include source hashes, outcomes, replay errors, and the planner benchmark.

The initial integration image passed all 16 tests. These cover repeatable reset snapshots, controller agreement, rollout isolation, button/lock transitions, insertion geometry, completed cube release, timeout/resume behavior, and LeRobot export/loading.

After the Tyro/YAML refactor, the rebuilt image passes all 23 tests, including configuration precedence, required fields, interpolation, and invalid-option checks. The existing example dataset still loads through the new CLI.

With the rollout viewer added, all 26 tests pass in Ubuntu/WSL. WSLg previews were checked through the PowerShell launcher for raw and exported rollouts, free/front/wrist cameras, paused playback, speed changes, terminal-state holding, and clean timed shutdown.

A later desktop check exposed WSLg's `[WARN:COPY MODE]` invisible-window failure. Restarting only the compositor restored shared-memory graphics transport, and the MuJoCo scene was visibly confirmed on the Windows desktop. The WSL launcher now supports `-RestartWslg`. Mesa initially used llvmpipe; selecting D3D12/NVIDIA reports `D3D12 (NVIDIA GeForce RTX 5090)`. Playback now restores simulator snapshots only when the frame changes and polls at 60 Hz. A short running-viewer sample used about 73% of one CPU core, compared with about seven cores under software rendering; this is not a sustained frame-rate benchmark.

Successful attempts were generated for tasks 1–5 in every manipulation variant:

| Environments | Goals with a successful attempt |
| --- | --- |
| cube-single, double, triple, quadruple, octuple | 25/25 |
| scene | 5/5 |
| puzzle-3x3, 4x4, 4x5, 4x6 | 20/20 |

Coverage was collected during development across several revisions, generally with five concurrent episodes, eight candidates, an eight-step horizon, one optimization iteration, and 64×64 images. It is not a success-rate estimate for the final source or random initial states. Rerun `validate` for the current implementation. All 56 checked development/sample outcomes agree with an independent upstream CPU goal check. The initial eight-cube task 4 timeout and failed scene insertion attempt remain available with failure labels.

## Example dataset

`E:\vla-ogbench\datasets\vla-ready` contains 1,192 frames at 20 Hz with 256×256 front/wrist RGB, state, actions, and instructions. It has three successes (cube placement, scene insertion, puzzle) and two failures (scene timeout and an intentional two-step timeout). Loading verifies `[16, 5]` action chunks and outcome filtering; replay reproduces all five outcomes.

```powershell
.\scripts\run.ps1 inspect --root /data/datasets/vla-ready --chunk-length 16 --outcome success
.\scripts\run.ps1 replay --root /data/datasets/vla-ready --episode 1
```

Replay does not guarantee identical contact trajectories. `max_qpos_error` mixes joint, position, and quaternion coordinates; quaternion signs can inflate this value without indicating an equivalent physical pose error. Use stored-state replay for inspecting the recorded trajectory.

## Contact-quality regression

The original example episode 1 (scene task 4, seed 2026) has five recorded states over the new non-pad contact threshold. Its deepest contact is 3.127 mm between the left gripper follower and drawer handle collision proxy. [Original audit](validation/drawer-contacts.json) keeps the original task-success label separate from contact validity.

A fresh attempt using the slower drawer skill and contact-constrained MPC succeeds in 306 frames. It is stored at `E:\vla-ogbench\raw\contact-guard-scene-v2` with 64?64 diagnostic RGB. GPU checks across all physics substeps and integrated endpoints report 0.562 mm peak non-pad penetration and 0.674 mm peak robot/environment penetration. The [independent recorded-state audit](validation/drawer-corrected-contacts.json) passes, with a 0.429 mm peak. These measurements use collision proxies and do not establish force realism, visual-mesh clearance, or success rates across seeds. The original exported example dataset remains unchanged.

The rebuilt image passes all 28 tests, including per-world contact reduction, ignored stale contact slots, allowed finger-pad/button contacts, and preservation of peak depths across substeps.

```powershell
.\scripts\view.ps1 -Root E:\vla-ogbench\raw\contact-guard-scene-v2 -Episode 0 --speed 0.5
```

## Cross-task contact validation

The contact sweep attempted all 50 goals across the ten manipulation variants, using five execution worlds, eight candidates, an eight-step horizon, one optimization iteration, and seeds 2026–2030. [Machine-readable audits](validation/contact-suite.json) retain source hashes, planner settings, failure reasons, and both recorded contact metrics and independent CPU audits. The sweep spans the revisions that added CPU checks and revised the window skill; it is not a success-rate estimate for the default planner or a test across many seeds.

| Environments | Goals with a contact-valid success |
| --- | --- |
| cube-single, double, triple, quadruple | 20/20 |
| cube-octuple | 1/5 |
| scene | 5/5 |
| puzzle-3x3, 4x4, 4x5, 4x6 | 20/20 |

All 55 successful recordings among 62 retained attempts pass the CPU contact audit. The four remaining eight-cube goals are failures: task 4 exhausted valid candidates, and tasks 2, 3, and 5 timed out at 1,500 steps. Earlier scene failures and an earlier eight-cube task 5 failure are retained. No failed attempt is counted as successful coverage.

The original window approach exhausted valid candidates on scene tasks 1 and 3. Greater approach clearance and slower grasping/translation produce contact-valid successes on all five scene goals. The same-seed opening regression succeeds in 161 frames with a 0.493 mm recorded peak and a [0.356 mm CPU audit peak](validation/window-corrected-contacts.json). Earlier 3×3 puzzle GPU metrics underreported pad penetration; current generation additionally checks CPU-restored observations and terminal success states, and the revised five-goal puzzle sweep passes independently.

The final image passes all 44 tests. These include contact roles in every manipulation model, reversed contact pairs, stale slots, per-world peaks, allowed button/pad contacts, CPU/GPU disagreement rejection, validation acceptance, and strict LeRobot selection excluding violations and missing contact evidence. Lint, formatting, and lockfile checks pass. WSLg playback of an image-free validation rollout opens and closes cleanly.

Raw attempts are under `E:\vla-ogbench\raw\contact-suite*`; revised scene recordings are in `contact-suite-scene-final`. `contact-suite-no-rgb` and `contact-suite-puzzle3-final` contain simulator states and actions without RGB and cannot be exported to LeRobot. Validation now skips RGB by default; use `--record-images` when images are needed. Normal generation still records RGB.

These checks bound robot/environment collision-proxy penetration. They do not establish force realism, self-collision safety, visual-mesh clearance, or correctness of unrecorded CPU substeps.

## Randomized joint targets

With `joint_target_noise=0.01`, the controller adds a seeded, constant offset of up to 0.01 radians to each arm joint target after IK. The [validation report](validation/joint-targets.json) covers all five goals in cube-single, scene, puzzle-3x3, and puzzle-4x6, using seeds 2026–2030. The small budget (five execution worlds, eight candidates, horizon eight, one iteration) produced 19 successes and one `invalid_candidates` failure on scene task 4, seed 2029. Retrying that same seed with 32 candidates and two iterations succeeded in 299 frames. All 20 successes passed independent CPU contact audits; the retry's peak recorded penetration was 0.572 mm. This checks 20 goals, not every environment or a success rate across many seeds. The image-free attempts are under `E:\vla-ogbench\raw\joint-target-validation-20261003` and `joint-target-scene-retry-20261003`.

All 52 tests pass, including post-IK target offsets, actuator limits, matching CEM candidate offsets, seed reproducibility, saved-state replay, and legacy snapshots without offsets. Lint, formatting, and lockfile checks pass.

## Trajectory diversity

The diversity implementation passes all 73 tests on the rebuilt Docker image. Checks cover independent seeded streams, grouped resets, keyframe annotations in every manipulation model, anchored interaction positions, timing floors, deterministic selection, and annotated/legacy LeRobot exports. Native Windows lint, formatting, and lockfile checks pass.

The combined recipe was exercised on all 50 registered goals with five execution worlds, four candidates, horizon four, one iteration, and seeds 2026-2030. It produced 45 successes, all passing independent CPU contact audits. Cube-quadruple task 5 and cube-octuple tasks 2-5 exhausted valid candidates and remain labeled failures. [Sweep results](validation/diversity-suite.json) retain each run's source hash, configuration, outcomes, and audits. This is a small-budget coverage check, not a success-rate estimate. The sweep used an earlier generation revision; the final image's tests and ablations are recorded separately.

[Factor ablations](validation/diversity-ablations.json) compare two same-reset variants per profile on cube-double task 2. All 14 attempts succeeded and passed CPU contact audits. The relative phase-aligned distance was 0.0545 for baseline, 0.2579 for grasp variation, and 0.2580 for combined settings. Order variation had little effect on this stacking goal's constrained choices. These two-variant measurements do not establish statistical significance or VLA training benefits; generation times include compilation and concurrent GPU load.

The [annotated example](validation/diversity-demo.json) contains four successful cube-single variants from the same saved initial state. All four pass independent CPU audits, with peak recorded penetration below 0.353 mm. Diversity selection exported two episodes and 250 frames with 32x32 diagnostic front/wrist RGB to `E:\vla-ogbench\datasets\diversity-demo-v1`. LeRobot loading verifies numeric annotations and `[8, 5]` action chunks; stored-state replay of episode 0 matches its recorded success. Raw attempts are in `diversity-demo-v1` under the raw data directory.

Optional handle half-turn grasps remain experimental and disabled in the recipe. Two scene task 4 attempts sampled the alternate drawer grasp and both exhausted valid candidates after 60 frames. Their provenance and audits are retained in the sweep report; neither is counted as a success.

## Workspace training integration

The root Docker launcher and timed WSLg viewer pass smoke checks. The shared Tyro/YAML SmolVLA launcher passes nine contract/provenance tests, plus lint, formatting, PowerShell syntax, and lockfile checks. It also serves the existing lighter training script with its explicit wrist-to-camera2 mapping.

A native Windows GPU run consumed the two annotated LeRobot demo episodes, completed one optimization step with batch size one and eight-action chunks, and saved a checkpoint with 18 state and five action dimensions. [Integration results](validation/stack-integration.json) retain the resolved configuration, dataset provenance hashes, source hashes, and checkpoint location. The checkpoint is under `E:\vla-ogbench\runs\stack-smoke-final-20261004`. This checks training compatibility, not task success or VLA performance; no hardware training or movement was performed.

## Planner benchmark

A five-tick cube smoke benchmark with the default budget (eight episodes, 128 candidates, horizon 16, three iterations) measured approximately 42,286 candidate control-rollout steps/s and 6.88 executed world steps/s, excluding image generation and export. Planning took about 1.15 seconds per batch tick; all execution worlds remained valid. Reported device usage was 4.69 GB, including other device allocations. This short benchmark does not measure sustained dataset throughput or demonstration quality.
