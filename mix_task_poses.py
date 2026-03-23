"""Mix a number of poses from two .task files into a single output .task file.

Takes the first N poses from task file A and the first M poses from task file B,
combines them (A then B by default, or interleaved), and writes one new .task file.
Uses the structure of the first task file as the template for the output.

Usage example:

    python mix_task_poses.py \\
        --task-a left_traj.task \\
        --task-b right_traj.task \\
        --from-a 10 \\
        --from-b 10 \\
        --output mixed.task

    # Interleave instead of concatenate:
    python mix_task_poses.py --task-a left.task --task-b right.task --from-a 5 --from-b 5 --output mixed.task --interleave
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_task_items(task_path: str | Path) -> list[dict[str, Any]]:
    """Load the parameter.parameterized_poses list from a Franka-style .task JSON."""
    task_path = Path(task_path)
    with open(task_path, "r") as f:
        data = json.load(f)
    parameter = data.get("parameter", {})
    items = parameter.get("parameterized_poses", [])
    if not isinstance(items, list):
        raise ValueError(f"'parameter.parameterized_poses' in {task_path} is not a list")
    return items


def build_output_task(
    template_path: str | Path,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a new .task JSON using template metadata and the given items."""
    template_path = Path(template_path)
    with open(template_path, "r") as f:
        data = json.load(f)
    if "parameter" not in data or not isinstance(data["parameter"], dict):
        data["parameter"] = {}
    data["parameter"]["parameterized_poses"] = items
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix poses from two .task files into one output .task file."
    )
    parser.add_argument(
        "--task-a",
        type=str,
        required=True,
        help="Path to first .task file.",
    )
    parser.add_argument(
        "--task-b",
        type=str,
        required=True,
        help="Path to second .task file.",
    )
    parser.add_argument(
        "--from-a",
        type=int,
        required=True,
        help="Number of poses to take from the first task file (from the start).",
    )
    parser.add_argument(
        "--from-b",
        type=int,
        required=True,
        help="Number of poses to take from the second task file (from the start).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .task file.",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Interleave poses (a0, b0, a1, b1, ...) instead of concatenating (a0..aN, b0..bM).",
    )
    parser.add_argument(
        "--template",
        choices=("a", "b"),
        default="a",
        help="Which task file to use as JSON template for the output (default: a).",
    )

    args = parser.parse_args()

    path_a = Path(args.task_a)
    path_b = Path(args.task_b)
    if not path_a.exists():
        raise FileNotFoundError(f"Task file not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Task file not found: {path_b}")

    items_a = load_task_items(path_a)
    items_b = load_task_items(path_b)

    n_a = args.from_a
    n_b = args.from_b
    if n_a > len(items_a):
        raise ValueError(f"--from-a {n_a} exceeds poses in {path_a} ({len(items_a)})")
    if n_b > len(items_b):
        raise ValueError(f"--from-b {n_b} exceeds poses in {path_b} ({len(items_b)})")

    selected_a = [items_a[i] for i in range(n_a)]
    selected_b = [items_b[i] for i in range(n_b)]

    if args.interleave:
        mixed: list[dict[str, Any]] = []
        for i in range(max(n_a, n_b)):
            if i < n_a:
                mixed.append(selected_a[i])
            if i < n_b:
                mixed.append(selected_b[i])
        combined = mixed
    else:
        combined = selected_a + selected_b

    template_path = path_a if args.template == "a" else path_b
    out_data = build_output_task(template_path, combined)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Wrote {len(combined)} poses to {out_path} ({n_a} from {path_a.name}, {n_b} from {path_b.name}).")


if __name__ == "__main__":
    main()
