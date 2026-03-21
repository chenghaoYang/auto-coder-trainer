"""Main CLI entry point for auto-coder-trainer.

Usage:
    act collect "coding agent training"
    act compose --atoms swe-fuse,entropy-rl
    act train recipe.json
    act report --experiment-id exp_001
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="act",
        description="Auto-Coder-Trainer: Research Operating System for Coding Agent Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # collect
    collect_parser = subparsers.add_parser("collect", help="Collect papers, projects, and methods")
    collect_parser.add_argument("query", type=str, help="Research query or topic")
    collect_parser.add_argument("--max-papers", type=int, default=20, help="Maximum papers to collect")
    collect_parser.add_argument("--max-repos", type=int, default=10, help="Maximum GitHub repos to collect")
    collect_parser.add_argument("--output", type=str, default="recipes/registry/", help="Output directory")

    # compose
    compose_parser = subparsers.add_parser("compose", help="Compose method atoms into a training recipe")
    compose_parser.add_argument("--atoms", type=str, required=True, help="Comma-separated method atom names")
    compose_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model")
    compose_parser.add_argument("--output", type=str, help="Output recipe path")

    # train
    train_parser = subparsers.add_parser("train", help="Execute a training experiment from a recipe")
    train_parser.add_argument("recipe", type=str, help="Path to recipe JSON file")
    train_parser.add_argument("--output-dir", type=str, default="outputs/", help="Output directory")
    train_parser.add_argument("--dry-run", action="store_true", help="Validate recipe without training")

    # report
    report_parser = subparsers.add_parser("report", help="Generate technical report from experiment results")
    report_parser.add_argument("--experiment-id", type=str, help="Experiment ID to report on")
    report_parser.add_argument("--recipe-id", type=str, help="Recipe ID to report on (all experiments)")
    report_parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    report_parser.add_argument("--output", type=str, default="reports/", help="Output directory")

    # status
    status_parser = subparsers.add_parser("status", help="Summarize tracked experiments and open tasks")
    status_parser.add_argument("--recipe-id", type=str, help="Limit the summary to a recipe")
    status_parser.add_argument("--open-only", action="store_true", help="Show only open tasks")
    status_parser.add_argument("--output", type=str, help="Optional path to save the status report")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommands
    if args.command == "collect":
        from cli.collect import run_collect
        run_collect(args)
    elif args.command == "compose":
        from cli.compose import run_compose
        run_compose(args)
    elif args.command == "train":
        from cli.train import run_train
        run_train(args)
    elif args.command == "report":
        from cli.report import run_report
        run_report(args)
    elif args.command == "status":
        from cli.status import run_status
        run_status(args)


if __name__ == "__main__":
    main()
