import argparse
import glob
import json
import os
from typing import Any, Dict, Iterator, List

import huggingface_hub
import pyarrow
import pyarrow.parquet

IMAGE = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
        ("benchmark", pyarrow.string()),
        ("scene", pyarrow.string()),
        ("instruction", pyarrow.string()),
        ("subtasks", pyarrow.list_(pyarrow.string())),
        ("n_goals", pyarrow.int32()),
        ("ordered", pyarrow.bool_()),
        ("step", pyarrow.int32()),
        ("num_steps", pyarrow.int32()),
        ("achieved", pyarrow.list_(pyarrow.bool_())),
        ("query_idx", pyarrow.int32()),
        ("label", pyarrow.bool_()),
        ("tags", pyarrow.list_(pyarrow.string())),
        # The scene contents the frame was cut under; check_reference fails a sample whose scene has
        # since changed what it places, which the wording fields cannot show.
        ("objects", pyarrow.list_(pyarrow.string())),
        ("arm", pyarrow.string()),
        ("source", pyarrow.string()),
        ("image_main", IMAGE),
        ("image_wrist", IMAGE),
    ]
)
SCALARS = (
    "id",
    "benchmark",
    "scene",
    "instruction",
    "subtasks",
    "n_goals",
    "ordered",
    "step",
    "num_steps",
    "achieved",
    "query_idx",
    "label",
    "tags",
    "objects",
    "arm",
    "source",
)
# One file per config, so a re-publish replaces it instead of leaving a second one behind: the
# scorer reads EVERY parquet under the config prefix and would silently pool two harvests.
REMOTE = "{config}/train-00000-of-00001.parquet"


def _image(path: str) -> Dict[str, Any]:
    if path is None or not os.path.isfile(path):
        return {"bytes": None, "path": None}
    return {"bytes": open(path, "rb").read(), "path": os.path.basename(path)}


def rows(sample_dirs: List[str]) -> Iterator[Dict[str, Any]]:
    for root in sample_dirs:
        for manifest in sorted(glob.glob(os.path.join(root, "*", "subtask_states.json"))):
            suite = os.path.dirname(manifest)
            for sample in json.load(open(manifest)):
                frames = {
                    camera: os.path.join(suite, name) for camera, name in (sample.get("frames") or {}).items()
                }
                yield {
                    **{key: sample[key] for key in SCALARS},
                    "image_main": _image(frames.get("main")),
                    "image_wrist": _image(frames.get("wrist")),
                }


def main():
    parser = argparse.ArgumentParser(description="Publish harvested samples as a dataset config.")
    parser.add_argument("--samples", nargs="+", required=True, help="harvest out-dirs to pool")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", default="subtask_states")
    parser.add_argument("--out", help="write the parquet here and skip the upload")
    args = parser.parse_args()

    table = pyarrow.Table.from_pylist(list(rows(args.samples)), schema=SCHEMA)
    if not table.num_rows:
        raise SystemExit(f"no samples under {args.samples}")
    positive = sum(table.column("label").to_pylist())
    print(f"{table.num_rows} samples, {positive} positive ({100 * positive / table.num_rows:.1f}%)")

    path = args.out or os.path.join("/tmp", REMOTE.format(config=args.config).replace("/", "_"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pyarrow.parquet.write_table(table, path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    if args.out:
        return

    huggingface_hub.upload_file(
        path_or_fileobj=path,
        path_in_repo=REMOTE.format(config=args.config),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"{args.config}: {table.num_rows} samples, {100 * positive / table.num_rows:.0f}% positive",
    )
    print(f"published {args.repo} :: {args.config}")


if __name__ == "__main__":
    main()
