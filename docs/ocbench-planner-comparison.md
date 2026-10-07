> Historical report: the OGBench package and its commands have been removed. See the [current workspace guide](../README.md) for OCBench and LIBERO.

# OCBench planner comparison

Inspected OCBench 1.0.1 at commit `e2cd2f72110b66bd65afab1b855d81ebc73aeacc` on 2026-10-06. Here, "ours" means the current cuRobo-initialized joint-spline generator, not the older CEM baseline. The recommended initial OCBench task is `block-double-task2-v0` (stack anywhere), which avoids a coordinate-specified goal.

## Architecture

| Property | Our spline generator | OCBench native block controller |
| --- | --- | --- |
| Path construction | cuRobo joint paths, bounded waypoint relaxation, quintic joint splines | Scripted Cartesian keyframes, cubic Hermite translation and eased rotation, online differential IK |
| Execution | Preplanned whole attempt; uniform or local retiming | Online scene-dependent subtask selection, tracking gates, grasp retries |
| GPU use | Batched cuRobo and MJWarp, with CPU geometry/quality checks | Warp controller, IK and physics across worlds |
| Collision treatment | Dense geometry checks and physics-substep penetration guards | No equivalent global collision optimizer or penetration audit found in the inspected native block path |
| Motion limits | Sampled joint velocity/acceleration/jerk checks and discrete action checks | Primitive duration/speed heuristics and tracking gates; no equivalent joint jerk-constrained retimer |
| Diversity | Explicit grasp proposals, approach and transport offsets, segment timing and speed strata | Randomized grasps, curved transport, closure timing, pauses, deliberate mistakes and retries |
| Ordinary failures | Continue through misses/drops; retain the failed attempt | Retry failed picks, reducing perturbations on retries |
| Actions | Seven absolute joint/gripper targets, 20 Hz | Seven normalized joint/gripper deltas, 50 Hz in the full environment |

OCBench's native implementation already provides much of the reactive behavior we were constructing. It also still contains task-specific controllers and contact primitives: switching does not eliminate task knowledge. Its GPU execution removes our expensive per-attempt cuRobo planning stage, making it a promising throughput improvement. That is an architectural inference, not a measured matched speedup.

Our checks are sampled numerical checks, not continuous collision or dynamics proofs. Conversely, native OCBench simulation and a native success label do not establish that a trajectory passes our quality criteria.

Sources: [block controller](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/controllers/block.py), [controller kernels](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/controllers/block_kernels.py), [cube primitives](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/primitives/cube_kernels.py), [primitive interpolation](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/primitives/primitive_kernels.py). Our implementation and prior measurements are documented in [spline generation](ogbench-spline-generation.md).

## Randomization and motion quality

The inspected full OCBench block controller samples a per-episode time scale from 0.7 to 1.5, grasp tilt magnitude up to 25 degrees, grasp position and yaw perturbations, transport side offsets of 4-14 cm, and gripper timing variation. Tilt magnitude is uniform, unlike our uniform-solid-angle cone. Its nominal yaw selection favors the nearest cube symmetry, with occasional quarter turns. Failed-grasp retries scale perturbations by `0.75 ** retry`; therefore realized grasp diversity depends on retry history.

The native controller pauses its plan clock at tracking gates with a 4 cm position tolerance. Its primitives deliberately stop at selected contact boundaries. Consequently it can still pause, and Cartesian spline smoothness does not guarantee smooth joint motion after IK or tracking. We should measure realized joint acceleration/jerk and inspect videos before describing either generator as more humanlike. Reactive corrections and retries are useful kinds of behavioral diversity that ours currently lacks.

The existing aggressive profile allows up to 35-degree tilt and explicitly chooses randomly among feasible grasp candidates. Thus OCBench is not automatically broader in every randomization dimension. Preserve distributions, sampled values, selected poses, retry counts, timing and realized outcomes in dataset annotations; seeds alone are insufficient for convenient filtering and analysis.

## Outcome labels are not interchangeable

For stack-anywhere, native OCBench checks block XY alignment within 4 cm and height within 3 cm of an ideal stack. It does not explicitly require gripper release, support contacts, uprightness or a one-second stable hold. Native termination can therefore precede completion of release/retreat. Its block health flag checks object workspace bounds, not our penetration audit. See [native success and health kernels](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/envs/block_kernels.py).

Our validated success additionally requires release, support, uprightness, tighter alignment and a stable hold. The datasets must retain separate `native_success`, audited success, physical validity, completion and termination-reason fields. A native healthy failure must not be exported as an audited physically valid failure without the additional checks.

Upstream collection resamples unhealthy attempts to fill its dataset quota. Our requested 500-attempt collection should instead preserve all raw attempts and filter exports afterward. Keep reset-seed train/validation assignment independent of outcome. See [upstream collection](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/impls/envs/streaming_data.py).

## Native pilot

A corrected 32-world pilot used environment seeds 82000-82031, oracle seeds 92000-92031 and a 2,500-step horizon on the RTX 5090. It produced **26/32 native successes**, with all 32 attempts passing native workspace health. Initialization took 8.35 seconds with cached kernels; execution and in-memory state/action recording took 39.01 seconds for 52,989 transitions. Compression, rendering, export and our physical audits are excluded. This is about 49 attempts/minute or 40 native successes/minute, not audited dataset throughput.

For context, our earlier, different OGBench 32-attempt trial produced 20 strict successes in 135.36 seconds after initialization, including validation and archive writing. Different environments, randomization, control rates, success criteria and measured work prevent treating these numbers as a planner speedup or success-rate comparison.

Pilot artifacts and the exact driver are under `C:\Users\steph\data\vla\ocbench-pilot-corrected` (`run.py`, `results.json`, `rollouts.npz`, `run.log`). Terminal states are captured before upstream parks completed worlds, and health is accumulated only for active worlds. The earlier `ocbench-pilot` archive has a faulty terminal-state/health measurement and must not be used for dataset export. The corrected archive passed finite-state/action and N+1-state checks, including terminal states remaining in the scene. Native solver iteration-budget warnings occurred; no additional penetration or convergence audit has been performed. Neither pilot is a finished LeRobot dataset.

## Migration boundaries

Use the native GPU oracle as the initial OCBench generation backend. Retain our raw-attempt provenance, independent quality audits, deferred batched rendering, LeRobot export and W&B workflow. Keep the existing planner available as a baseline while those integrations are tested.

The native seven-dimensional action vector is not compatible with existing absolute-target ACT checkpoints. Give it a distinct action profile with delta scales and control timestep, and reject mismatched checkpoints even when tensor shapes match. Preserve proprioceptive observations without including privileged object state in policy inputs.

OCBench's current native pixel-observation path loops through worlds using CPU-side rendering; GPU simulation does not imply batched camera rendering. Reuse our MJWarp renderer after checking camera placement and geometry compatibility. See [native environment implementation](https://github.com/seohongpark/ocbench/blob/e2cd2f72110b66bd65afab1b855d81ebc73aeacc/ocbench/mjwarp/envs/manipulation.py).

The 500-attempt OGBench collection was stopped at zero completed attempts when the user requested OCBench. Its completed 16-attempt pilot and exports remain available. The OCBench dataset/action/evaluation bridge is now implemented; see the [integration report](ocbench-workflow.md) for validation and run status.

For a matched comparison, run both generators on the same OCBench resets and goal, with the same controller interface, termination policy, physical audit and rendering settings. Report warmup separately, attempts and audited successes per minute, valid failures per minute, motion duration, pauses, realized acceleration/jerk and diversity conditioned on outcome. Existing OGBench and OCBench pilot percentages cannot establish which planner is better.
