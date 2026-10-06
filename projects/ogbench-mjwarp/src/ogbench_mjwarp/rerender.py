"""Render future state-only recordings in batches without changing their source."""

import json
import shutil
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .config import PlannerConfig
from .environment import BatchEnvironment
from .io import episode_metadata, write_json
from .tasks import image_shape, make_env


def rerender(source, output, batch_size=32, episode_ids=None):
    started = time.perf_counter()
    source, output = Path(source), Path(output)
    run = json.loads((source / "run.json").read_text())
    if run.get("format") not in ("ogbench-rollouts-2", "ogbench-rollouts-3"):
        raise ValueError(
            "Rerender supports new v2 raw recordings only; no legacy migration"
        )
    if batch_size < 1 or output.exists():
        raise ValueError("Need a positive batch size and a fresh output directory")
    rows = episode_metadata(source)
    if episode_ids is not None:
        rows = [row for row in rows if row["episode_id"] in episode_ids]
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
                    env,
                    len(batch),
                    PlannerConfig(**(run["planner"] | {"episodes": len(batch)})),
                )
                arrays = []
                for row in batch:
                    with np.load(
                        source / row["archive"], allow_pickle=False
                    ) as archive:
                        arrays.append({key: archive[key] for key in archive.files})
                # Stream image members: 32 entire RGB videos would exceed host RAM.
                with ExitStack() as stack:
                    streams = []
                    for row in batch:
                        views = {}
                        for view in ("front", "wrist"):
                            path = output / f"{row['archive']}.{view}.tmp"
                            archive = stack.enter_context(
                                zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
                            )
                            stream = stack.enter_context(
                                archive.open(f"{view}.npy", "w", force_zip64=True)
                            )
                            np.lib.format.write_array_header_1_0(
                                stream,
                                {
                                    "descr": "|u1",
                                    "fortran_order": False,
                                    "shape": (
                                        row["length"],
                                        *image_shape(run["image_size"]),
                                        3,
                                    ),
                                },
                            )
                            views[view] = stream
                        streams.append(views)
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
                                for view, stream in streams[world].items():
                                    stream.write(rendered[view][world].tobytes())
                profile = sim.renderer.profile
                for row, data in zip(batch, arrays, strict=True):
                    path = output / row["archive"]
                    with path.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, **data)
                    with zipfile.ZipFile(
                        path.with_suffix(".npz.tmp"), "a", zipfile.ZIP_DEFLATED
                    ) as destination:
                        for view in ("front", "wrist"):
                            temp = output / f"{row['archive']}.{view}.tmp"
                            with (
                                zipfile.ZipFile(temp) as origin,
                                origin.open(f"{view}.npy") as reader,
                                destination.open(
                                    f"{view}.npy", "w", force_zip64=True
                                ) as writer,
                            ):
                                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                            temp.unlink()
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
    result = {
        "output": str(output),
        "episodes": len(rows),
        "render_and_archive_seconds": time.perf_counter() - started,
    }
    write_json(output / "render_metrics.json", result)
    return result
