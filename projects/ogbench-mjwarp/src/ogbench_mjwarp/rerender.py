"""Render future state-only recordings in batches without changing their source."""

import json
from pathlib import Path

import numpy as np

from .config import PlannerConfig
from .environment import BatchEnvironment
from .io import episode_metadata, write_json
from .tasks import make_env


def rerender(source, output, batch_size=32):
    source, output = Path(source), Path(output)
    run = json.loads((source / "run.json").read_text())
    if run.get("format") != "ogbench-rollouts-2":
        raise ValueError(
            "Rerender supports new v2 raw recordings only; no legacy migration"
        )
    if batch_size < 1 or output.exists():
        raise ValueError("Need a positive batch size and a fresh output directory")
    rows = episode_metadata(source)
    if not rows or not any(row["length"] for row in rows):
        raise ValueError("No nonempty completed recordings")
    output.mkdir(parents=True)
    try:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            env = make_env(
                run["env_id"], batch[0]["seed"], batch[0]["task_id"], run["image_size"]
            )
            try:
                sim = BatchEnvironment(
                    env, len(batch), PlannerConfig(episodes=len(batch))
                )
                arrays = []
                for row in batch:
                    with np.load(
                        source / row["archive"], allow_pickle=False
                    ) as archive:
                        arrays.append({key: archive[key] for key in archive.files})
                images = [{view: [] for view in ("front", "wrist")} for _ in batch]
                for frame in range(max(row["length"] for row in batch)):
                    sim.restore(
                        {
                            key[4:]: np.stack(
                                [
                                    data[key][min(frame, row["length"])]
                                    for data, row in zip(arrays, batch, strict=True)
                                ]
                            )
                            for key in arrays[0]
                            if key.startswith("sim/")
                        }
                    )
                    rendered = sim.render_batch()
                    for world, row in enumerate(batch):
                        if frame < row["length"]:
                            for view in images[world]:
                                images[world][view].append(rendered[view][world].copy())
                profile = sim.renderer.profile
                for row, data, frames in zip(batch, arrays, images, strict=True):
                    data.update(
                        {view: np.asarray(value) for view, value in frames.items()}
                    )
                    path = output / row["archive"]
                    with path.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, **data)
                    path.with_suffix(".npz.tmp").replace(path)
                    write_json(
                        output / f"episode-{row['episode_id']:06d}.json",
                        row | {"record_images": True, "rendering": profile},
                    )
            finally:
                env.close()
        write_json(
            output / "run.json", run | {"record_images": True, "rendering": profile}
        )
    except BaseException:
        write_json(output / "INCOMPLETE.json", {"source": str(source)})
        raise
    return {"output": str(output), "episodes": len(rows)}
