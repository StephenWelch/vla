> Historical report: the OGBench package and its commands have been removed. See the [current workspace guide](../README.md) for OCBench and LIBERO.

> Archived v1 results. The datasets and OpenGL recording path are retired; raw recordings and historical checkpoints are retained. Use the v2 training workflow for new experiments.

# OGBench ACT and SmolVLA pilots

Run on 2026-10-04 with an RTX 5090 (32 GB), native Windows training, LeRobot 0.6.1 and PyTorch 2.11.0+cu130. Closed-loop evaluation uses the isolated Ubuntu/WSL environment with PyTorch 2.11.0 and the same GPU. The requested scope was pilot training on existing data. The initial ACT/pi0.5 comparison was changed to ACT/SmolVLA at the user's request; pi0.5 recovery notes are retained below.

## Data and interface

Both recipes use `E:\vla-ogbench\datasets\diversity-demo-v1`: two successful, contact-valid cube-single task 1 episodes, 250 frames at 20 Hz, front/wrist RGB at 32x32, 18 state values, and five normalized OGBench action values. These are diagnostic images. Both episodes share the same initial state and goal; there is no held-out split in this pilot.

The dataset manifest retains the generator's randomization settings, independent seeds, sampled values, and per-frame skill/phase annotations. Training consumes observations/actions; the annotations are not policy inputs. The shared launcher records configuration, dataset metadata hashes, camera mapping, native training arguments, status, and full training logs. `train-smolvla.py` remains a compatibility entry point to `train-policy.py`.

## Results

| Policy | Status | Updates | Batch | Action chunk |
| --- | --- | --- | --- | --- |
| ACT | Completed | 500 | 8 | 16 |
| SmolVLA | Completed | 500 | 8 | 16 |
| pi0.5 | Cancelled when the user selected SmolVLA instead | 0 | Planned 1 | 16 |

ACT initializes its transformer/VAE from scratch and uses the default ImageNet-pretrained ResNet-18 backbone. It uses the original front/wrist feature names and the dataset's 18-state/5-action dimensions. ACT does not condition on language, so this single-goal pilot is suitable for testing its data contract but does not test instruction following.

The final logged ACT training averages are total loss **2.530**, normalized L1 **0.426**, and KL **0.210** (`total = L1 + 10 * KL`). These are averages over the last logging interval, not held-out metrics. Training completed in approximately 64 seconds, excluding imports/model setup; this short run is not a throughput benchmark.

Reloading the final checkpoint and predicting the first action of a fresh chunk on 20 evenly spaced frames per episode gives:

| In-sample prediction metric | ACT | SmolVLA | Zero-action baseline |
| --- | --- | --- | --- |
| Mean absolute error | 0.1740 | 0.1786 | 0.3483 |
| Mean squared error | 0.06516 | 0.06641 | 0.23013 |

Metrics use the original normalized five-element commands after checkpoint postprocessing. This is a recorded-frame fit check on training data. It does not measure simulator success, contact quality under the learned policy, or generalization. Do not compare ACT's training loss directly with SmolVLA's flow-matching loss.

The ACT run is at `E:\vla-ogbench\runs\act-pilot-20261004`, with checkpoints at steps 250 and 500. The final policy is `checkpoints\000500\pretrained_model`; `experiment.json`, `train.log`, and `offline-fit.json` retain the detailed evidence. [Machine-readable results](../projects/ogbench-mjwarp/validation/policy-pilots.json) include provenance and the checkpoint hash.

## SmolVLA pilot

`configs/train-ogbench-smolvla-pilot.yaml` fine-tunes the existing local `E:\vla-smolvla\smolvla_base` checkpoint on the same two episodes, with the same seed, batch size, 500 updates, and 16-action chunks as ACT. Front/wrist map to camera1/camera2; state and action dimensions are rebound to 18 and five. The inherited SmolVLA settings freeze the vision encoder and train the action expert/state projection. Only the final checkpoint is saved.

The final logged SmolVLA loss is **0.596**, averaged over the last logging interval. The update loop took approximately 146 seconds, excluding imports/model setup and checkpoint saving. The saved checkpoint was reloaded successfully for the 40-frame fit check above. These results establish that both training paths work on this dataset; the tiny in-sample comparison does not establish which architecture performs better.

The SmolVLA run is at `E:\vla-ogbench\runs\smolvla-pilot-20261004`. Its final policy is `checkpoints\000500\pretrained_model`; `experiment.json`, `train.log`, and `offline-fit.json` record configuration, provenance, training, and predictions. The checkpoint retains an inherited camera3 feature declaration, but only the available front/wrist images are supplied (`empty_cameras=0`).

```powershell
uv run python scripts\train-policy.py --config configs\train-ogbench-smolvla-pilot.yaml --output E:/vla-ogbench/runs/smolvla-pilot-new
uv run python scripts\evaluate-policy.py --dataset E:/vla-ogbench/datasets/diversity-demo-v1 --checkpoint E:/vla-ogbench/runs/smolvla-pilot-new/checkpoints/000500/pretrained_model --output E:/vla-ogbench/runs/smolvla-pilot-new/offline-fit.json --hf-home E:/vla-smolvla/hf-cache
```

## Reproduce ACT

From the workspace root:

```powershell
uv sync --frozen
uv run python scripts\train-policy.py --config configs\train-ogbench-act.yaml --output E:/vla-ogbench/runs/act-pilot-new
uv run python scripts\evaluate-policy.py --dataset E:/vla-ogbench/datasets/diversity-demo-v1 --checkpoint E:/vla-ogbench/runs/act-pilot-new/checkpoints/000500/pretrained_model --output E:/vla-ogbench/runs/act-pilot-new/offline-fit.json --hf-home E:/vla-smolvla/hf-cache
```

Training requires a new output path and preserves existing runs. YAML defaults are overridden by Tyro CLI flags. `--overrides` accepts native LeRobot hyperparameters such as `policy.chunk_size=16`; dataset paths, camera mapping and training budgets use the typed fields. The launcher passes thirteen contract/provenance tests, plus lint, formatting and lockfile checks.

## Closed-loop OGBench evaluation

`ogbench_mjwarp.lerobot_env` registers `OGBenchEnvConfig` as `env.type=ogbench`, following LeRobot's [environment extension interface](https://huggingface.co/docs/lerobot/main/adding_benchmarks). `create_envs()` returns Gymnasium vector environments backed by the existing `BatchEnvironment`. Physics and policy inference batch on the GPU; front/wrist cameras render sequentially through the shared CPU MuJoCo model. No installed LeRobot files are modified.

The adapter supplies native 20 Hz control, 18-state observations, front/wrist uint8 images, task instructions, and five commands clipped to [-1, 1]. LeRobot's saved preprocessors retain ACT's original camera names and SmolVLA's camera1/camera2 mapping, dataset normalization, and action postprocessing. The evaluator calls LeRobot's `eval_policy()` for resets, action-chunk execution, metrics, and video writing. Finished worlds freeze until the next explicit rollout reset.

Evaluation uses `cube-single-v0`, task 1, with 32x32 images and a **250-step (12.5-second) horizon**, roughly twice the demonstration length. Both checkpoints are evaluated on the training reset (seed 2026) and ten new reset seeds (2027–2036), with five concurrent worlds in the latter run. These are new reset seeds for the same predefined task and goal, not a held-out instruction/task benchmark. No controller noise is added during evaluation.

LeRobot success requires task completion, finite/valid physics, and contact validity throughout the episode. As in generation, contact checks cover physics substeps and the integrated endpoint: at most 1 mm of nonpad penetration and 3 mm of overall robot penetration, using the existing pad/button contact roles. Reports also retain raw task completion, sticky contact validity, penetration peaks, termination/timeout, and episode seeds. These collision checks do not guarantee absence of visual-mesh overlap.

| Policy | Training reset success | New-reset success | New-reset contact valid |
| --- | --- | --- | --- |
| ACT | 0/1 | 0/10 | 9/10 |
| SmolVLA | 0/1 | 0/10 | 9/10 |

Both policies timed out in all eleven episodes with valid physics. ACT reset 2034 exceeded the nonpad threshold (1.086 mm), as did SmolVLA reset 2028 (1.781 mm). Both passed contacts on the training reset. No episode reached the task goal, so raw task success and contact-valid success agree. Recorded-frame fit has not translated into task success for these 500-update pilots.

[Machine-readable evaluation results](../projects/ogbench-mjwarp/validation/policy-evaluation.json) retain per-episode outcomes, seeds, checkpoint hashes, configuration, source/version provenance, and video/recording checks.

Runs are under each policy's `E:\vla-ogbench\runs\<policy>-pilot-20261004` directory: ACT's training-reset run is `eval-seen-v2`, SmolVLA's is `eval-seen`, and both new-reset runs are `eval-heldout`. Each contains `eval_info.json` and `task-1/videos/eval_episode_*.mp4` (front/wrist views side by side). Logs are adjacent to the evaluation directory. The initial ACT `eval-seen` run completed its rollout but failed to serialize NumPy goal metadata; it was corrected and rerun, and is excluded from results.

### Reproduce from PowerShell

The installed evaluation Python is `$HOME/.venvs/vla-policies/bin/python` inside WSL. To prepare that environment from WSL, with `uv` installed:

```bash
cd /mnt/c/Users/steph/code/vla
uv venv --python 3.12 "$HOME/.venvs/vla-policies"  # Only when creating a new environment.
uv pip install --python "$HOME/.venvs/vla-policies/bin/python" -e './projects/ogbench-mjwarp[evaluation]'
```

The PowerShell launcher passes the existing local authentication file by path when available; it does not copy or log credentials. YAML paths use WSL mount paths. Use `-Python` to select another prepared Linux environment.

```powershell
.\scripts\eval-ogbench.ps1 -Config configs/eval-ogbench-act.yaml --output /mnt/e/vla-ogbench/runs/act-eval-new
.\scripts\eval-ogbench.ps1 -Config configs/eval-ogbench-smolvla.yaml --output /mnt/e/vla-ogbench/runs/smolvla-eval-new
# Training-reset check:
.\scripts\eval-ogbench.ps1 -Config configs/eval-ogbench-act.yaml --episodes 1 --batch-size 1 --seed 2026 --output /mnt/e/vla-ogbench/runs/act-seen-new
```

Outputs must be new directories. `--task-ids`, `--max-steps`, `--batch-size`, `--episodes`, and `--videos` override YAML defaults. Other manipulation families use `--env scene-v0` or `--env puzzle-3x3-v0`; these pilot checkpoints have not been evaluated or trained on those tasks.

### Native LeRobot CLI

The registration wrapper also works with LeRobot's original evaluation entry point. From WSL, with the evaluation environment active:

```bash
MUJOCO_GL=egl python scripts/lerobot-eval-ogbench.py \
  --policy.path=/mnt/e/vla-ogbench/runs/act-pilot-20261004/checkpoints/000500/pretrained_model \
  --policy.device=cuda --env.type=ogbench --env.task=cube-single-v0 \
  --env.image_size=32 --env.max_steps=250 \
  --eval.batch_size=5 --eval.n_episodes=10 --seed=2027 \
  --output_dir=/mnt/e/vla-ogbench/runs/act-native-eval-new
```

For SmolVLA, the native CLI additionally needs `--rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'`, since it overrides the checkpoint's rename processor. The typed launcher preserves the saved mapping automatically and adds contact diagnostics to the standard LeRobot results. A one-step ACT run through the native CLI completed successfully; it is a compatibility smoke test, excluded from task-performance results.

The native CLI also accepts `--eval.recording=true` to record observations/actions as a LeRobot v3 dataset instead of summary videos. A two-step ACT recording at `eval-recording-smoke/recordings/cube-single-v0_1` was saved and reloaded with PyAV, confirming the 18-state/5-action contract and both camera streams. The adapter exposes a shared state alias for the recorder's feature names while retaining LeRobot's standard `agent_pos` preprocessing. Recording diagnostics emitted an unavailable-TorchCodec warning in WSL; PyAV decoding succeeded. Neither recording smoke run is included in the success-rate table.

Validation passed 65 non-GPU subproject tests plus the new two-world GPU adapter test. Tests cover sticky contact rejection, success/timeout handling, finished-world freezing, action validation, deterministic resets, state/image shapes, and LeRobot environment registration. Lint and formatting checks also pass. The evaluated policies are the original 500-step pilot checkpoints; evaluation does not train or move hardware.

## Evaluation cleanup validation

Evaluation setup, task execution, and reporting now have separate responsibilities. `scripts/policy_utils.py` shares YAML/Tyro parsing, authentication-aware cache setup, and saved-policy loading across the policy tools. Shared simulator metadata utilities serialize paths directly and collect version provenance without importing the planner. The adapter uses masked restore and cached terminal images, while returned observations retain independent image buffers.

All 94 subproject tests (including GPU tests) and 13 training-launcher tests passed. A six-tick, two-world trace with different finish times matched all 144 state/image/outcome/contact arrays within 1e-6 and produced identical episode records. ACT's complete 40-frame offline prediction report matched the original exactly. ACT and SmolVLA each completed an eight-step, three-episode check with two concurrent worlds, exercising partial-batch trimming and video output. Native LeRobot recording saved two frames and reloaded successfully with PyAV.

Cleanup checks are separate from the task-performance results: artifacts are under `E:\vla-ogbench\runs\eval-cleanup`, with [machine-readable validation](../projects/ogbench-mjwarp/validation/evaluation-cleanup.json). Existing checkpoints, datasets, pilot reports, evaluation budgets, and contact thresholds were preserved. These short checks do not add evidence of policy task success.

## Cancelled pi0.5 attempt

The [official pi0.5 instructions](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx) require access to the gated [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224). Access now succeeds using the user's existing login. The launcher preserves the default authentication file when switching `HF_HOME` to the E: model cache; it never copies or logs credentials.

The 14.47 GB weights were downloaded at `E:\vla-ogbench\models\pi05_base`, pinned to revision `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`. Their SHA-256 was `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`; standalone CPU loading read all 812 tensors. The HF CLI finished the download but raised `click.exceptions.Exit: 0`; the shared Python API downloader avoids that CLI failure and records `source.json`.

Two native Windows attempts failed before the first optimization step with exit code `3221225477` (`0xC0000005`). Fault-handler output locates the access violation in `torch.storage.__getitem__` during `safetensors.load_file`. Both `.experiment.json` and `.train.log` sidecars remain under `E:\vla-ogbench\runs` for `pi05-pilot-20261004` and `pi05-pilot-20261004-retry1`. The retry limited CPU threads and enabled synchronous CUDA execution; it reproduced the failure. Its underlying cause has not been established.

E: then had no free space, and installing a WSL environment on that drive failed with `No space left on device`. An isolated fallback environment was successfully installed at `/home/stephen/.venvs/vla-policies` in Ubuntu/WSL, with LeRobot 0.6.1, Torch 2.11.0, and Python 3.12.3. It loaded the pretrained weights with all keys accepted. The user selected SmolVLA instead, so this smoke process was terminated before its first optimization step; no pi0.5 checkpoint was saved.

Only the newly downloaded pi0.5 `model.safetensors` was removed to reclaim approximately 14.47 GB for SmolVLA training. Model revision metadata, recipes, and failed-attempt logs were preserved. Existing datasets and other model checkpoints were not removed. The cancelled WSL run's sidecars are under `/home/stephen/vla-runs/pi05-smoke-20261004`.

To download assets on another machine after accepting the tokenizer license and authenticating locally:

```powershell
uv run hf auth login
uv run python scripts\download-policy.py --repo-id lerobot/pi05_base --revision b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --output E:/vla-ogbench/models/pi05_base --hf-home E:/vla-smolvla/hf-cache --tokenizer-repo google/paligemma-3b-pt-224
```

The recipe maps front/wrist to `base_0_rgb`/`left_wrist_0_rgb`, uses bfloat16, gradient checkpointing, and freezes the VLM/vision encoder while training the action expert. It preserves the base model's padded state/action capacity and uses the dataset's quantile statistics. Use a new output path for every attempt; completed runs and failed sidecars are preserved.

Before a meaningful architecture comparison, generate more 256x256 demonstrations across initial states/tasks and hold out entire reset seeds/scenarios, keeping variants of one reset in the same split. The small closed-loop checks above do not establish generalization or architecture superiority. No hardware movement was performed.
