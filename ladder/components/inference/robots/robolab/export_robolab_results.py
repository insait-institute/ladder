import argparse
import json
import os

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a ladder-runner campaign (episode_log.yaml) to RoboLab's episode_results.jsonl."
    )
    parser.add_argument("--episode-log", required=True, help="episode_log.yaml from the ladder runner.")
    parser.add_argument("--output-dir", required=True, help="Directory for episode_results.jsonl.")
    parser.add_argument("--policy", default="ladder_vlam", help="Policy label stamped into records.")
    args = parser.parse_args()

    with open(args.episode_log) as f:
        episode_log = yaml.safe_load(f) or []

    # Batches arrive in order per task; number them 0.. like RoboLab's run index.
    batch_to_run = {}
    per_task_runs = {}
    records = []
    for entry in episode_log:
        task = entry["env_name"]
        batch_key = (task, entry.get("batch", entry.get("episode")))
        if batch_key not in batch_to_run:
            batch_to_run[batch_key] = per_task_runs.get(task, 0)
            per_task_runs[task] = batch_to_run[batch_key] + 1
        run_index = batch_to_run[batch_key]
        records.append(
            {
                "task_name": task,
                "run_name": f"{task}_{run_index}",
                "run": run_index,
                "env_id": entry.get("batch_env_index", 0),
                "policy": args.policy,
                "instruction": entry.get("instruction", ""),
                "success": bool(entry["success"]),
                "episode_step": entry["num_steps"],
                "metrics": entry.get("metrics", {}),
                "events": {},
                "timing": {},
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "episode_results.jsonl")
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
