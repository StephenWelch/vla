"""Phase-aligned trajectory comparison and deterministic dataset selection."""

import time
from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from .io import episode_metadata, write_json
from .randomization import initial_state_fingerprint


def trajectory_descriptors(rows, samples=8):
    """Represent realized paths by skill/phase, with wrapped angles and duration."""
    if samples < 2:
        raise ValueError("samples must be >=2")
    descriptors = []
    for root, row in rows:
        with np.load(Path(root) / row["archive"], allow_pickle=False) as archive:
            state = archive["state"]
            features = np.column_stack(
                (
                    np.sin(state[:, :6]),
                    np.cos(state[:, :6]),
                    state[:, 12:15] / 0.5,
                    np.sin(state[:, 15]),
                    np.cos(state[:, 15]),
                    state[:, 16],
                )
            )
            available = (
                row.get("randomization", {}).get("available") is True
                and "annotation/skill_id" in archive
            )
            skill = (
                archive["annotation/skill_id"]
                if available
                else np.zeros(len(state), dtype=int)
            )
            phase = (
                archive["annotation/phase_id"]
                if available
                else np.zeros(len(state), dtype=int)
            )
            events = row.get("randomization", {}).get("skills", [])
            boundaries = np.r_[
                0,
                np.flatnonzero((np.diff(skill) != 0) | (np.diff(phase) != 0)) + 1,
                len(state),
            ]
            descriptor, occurrences, order = {}, defaultdict(int), []
            for start, end in pairwise(boundaries):
                event = (
                    events[int(skill[start])]
                    if available
                    else {"kind": "unknown", "index": -1, "phase_names": ["unknown"]}
                )
                key = (
                    event["kind"],
                    event.get("index", -1),
                    event["phase_names"][int(phase[start])],
                )
                occurrence = occurrences[key]
                occurrences[key] += 1
                path = features[start:end]
                indices = np.linspace(0, len(path) - 1, samples)
                values = np.column_stack(
                    [
                        np.interp(indices, np.arange(len(path)), path[:, column])
                        for column in range(path.shape[1])
                    ]
                )
                descriptor[(*key, occurrence)] = np.r_[
                    1.0, (end - start) / row["fps"], values.ravel()
                ]
                if not order or order[-1] != int(skill[start]):
                    order.append(int(skill[start]))
            # Encode executed object/skill order separately from phase-normalized paths.
            sequence = (
                [
                    events[i].get("index", -1)
                    + {
                        "cube": 0,
                        "button": 100,
                        "drawer": 200,
                        "window": 300,
                        "none": 400,
                    }.get(events[i]["kind"], 500)
                    for i in order
                ]
                if available
                else [-1]
            )
            descriptor[("order", -1, "sequence", 0)] = (
                np.interp(
                    np.linspace(0, len(sequence) - 1, samples),
                    np.arange(len(sequence)),
                    sequence,
                )
                / 100
            )
            descriptors.append(descriptor)
    keys = sorted({key for descriptor in descriptors for key in descriptor})
    sizes = {key: next(len(d[key]) for d in descriptors if key in d) for key in keys}
    return np.array(
        [
            np.concatenate([d.get(key, np.zeros(sizes[key])) for key in keys])
            for d in descriptors
        ]
    )


def diverse_selection(rows, count):
    """Select successful contact-valid recordings, balanced by environment/task."""
    if count < 1:
        raise ValueError("diverse_per_task must be positive")
    groups = defaultdict(list)
    for root, row in rows:
        if (
            row["outcome"] == "success"
            and row.get("contact_quality", {}).get("valid") is True
        ):
            groups[(row["env_id"], row["task_id"])].append((root, row))
    selected = []
    for group in groups.values():
        vectors = trajectory_descriptors(group)
        # Start near the group center; expand using farthest-point sampling.
        first = int(np.argmin(np.square(vectors - vectors.mean(0)).mean(1)))
        kept = [first]
        distances = np.square(vectors - vectors[first]).mean(1)
        while len(kept) < min(count, len(group)):
            distances[kept] = -1
            winner = int(distances.argmax())
            kept.append(winner)
            distances = np.minimum(
                distances, np.square(vectors - vectors[winner]).mean(1)
            )
        selected.extend(group[i] for i in kept)
    return selected


def measure_diversity(source, samples=8):
    """Compare successful contact-valid variants sharing the same reset seed."""
    sources = (
        [Path(source)] if isinstance(source, (str, Path)) else [Path(p) for p in source]
    )
    groups = defaultdict(list)
    for root in sources:
        for row in episode_metadata(root, "success", True):
            if row["length"]:
                fingerprint = row.get("randomization", {}).get(
                    "initial_state_fingerprint"
                )
                if fingerprint is None:
                    with np.load(root / row["archive"], allow_pickle=False) as archive:
                        if "sim/qpos" in archive and "sim/qvel" in archive:
                            fingerprint = initial_state_fingerprint(
                                {
                                    key[4:]: archive[key][0]
                                    for key in archive.files
                                    if key.startswith("sim/")
                                }
                            )
                groups[
                    (row["env_id"], row["task_id"], row["seed"], fingerprint)
                ].append((root, row))
    report = []
    for (env, task, seed, fingerprint), rows in groups.items():
        vectors = trajectory_descriptors(rows, samples)
        distances = [
            float(np.sqrt(np.square(vectors[i] - vectors[j]).mean()))
            for i in range(len(rows))
            for j in range(i)
        ]
        report.append(
            {
                "env": env,
                "task_id": task,
                "environment_seed": seed,
                "initial_state_fingerprint": fingerprint,
                "reset_identity": "saved_state_fingerprint"
                if fingerprint
                else "seed_only",
                "episodes": len(rows),
                "pairs": len(distances),
                "mean_distance": float(np.mean(distances)) if distances else None,
                "min_distance": min(distances) if distances else None,
                "max_distance": max(distances) if distances else None,
            }
        )
    return {
        "metric": "RMS normalized phase-aligned paths, segment duration and skill order; angles represented as sine/cosine",
        "samples_per_phase": samples,
        "groups": report,
    }


def run_ablations(
    output,
    env_id,
    task_id,
    variants,
    seed,
    planner,
    randomization,
    size=32,
    record_images=False,
):
    """Compare each factor and their combination on identical reset states."""
    from .contacts import audit_contacts
    from .recording import generate

    if variants < 2:
        raise ValueError("Ablations require at least two variants")
    output = Path(output)
    baseline = replace(
        randomization,
        variants_per_reset=variants,
        order=False,
        cube_grasps=False,
        handle_grasps=False,
        position_noise=0.0,
        yaw_noise=0.0,
        duration_scale_min=1.0,
        duration_scale_max=1.0,
    )
    profiles = {
        "baseline": baseline,
        "order": replace(baseline, order=randomization.order),
        "grasp": replace(
            baseline,
            cube_grasps=randomization.cube_grasps,
            handle_grasps=randomization.handle_grasps,
        ),
        "path": replace(
            baseline,
            position_noise=randomization.position_noise,
            yaw_noise=randomization.yaw_noise,
        ),
        "timing": replace(
            baseline,
            duration_scale_min=randomization.duration_scale_min,
            duration_scale_max=randomization.duration_scale_max,
        ),
        "joint_targets": baseline,
        "combined": replace(randomization, variants_per_reset=variants),
    }
    report = {
        "env": env_id,
        "task_id": task_id,
        "environment_seed": seed,
        "variants": variants,
        "profiles": [],
    }
    for name, config in profiles.items():
        search = (
            planner
            if name in ("joint_targets", "combined")
            else replace(planner, joint_target_noise=0.0)
        )
        start = time.perf_counter()
        root = output / name
        summary = generate(
            root,
            env_id,
            variants,
            [task_id],
            seed,
            search,
            size=size,
            record_images=record_images,
            randomization=config,
        )
        elapsed = time.perf_counter() - start
        audits = [
            audit_contacts(
                root,
                row["episode_id"],
                search.max_nonpad_penetration,
                search.max_penetration,
            )
            for row in episode_metadata(root)
        ]
        report["profiles"].append(
            {
                "name": name,
                "summary": summary,
                "generation_seconds": elapsed,
                "contact_audits": audits,
                "all_successes_contact_valid": all(
                    row["contact_valid"]
                    for row in audits
                    if row["recorded_outcome"] == "success"
                ),
                "diversity": measure_diversity(root),
            }
        )
        write_json(output / "ablation.json", report)
    return report
