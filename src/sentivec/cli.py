"""Local commands; never publishes a site, opens a tunnel, or pushes a repository."""

import argparse
from pathlib import Path

from .config import profile


def main():
    parser = argparse.ArgumentParser(description="SentiVec reproducible research")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the cited Kaggle dataset")
    download.add_argument("--destination", default="data/raw")
    run = commands.add_parser("run", help="Execute the neural research protocol")
    run.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="pilot")
    run.add_argument("--imdb-path", default="data/raw/IMDB Dataset.csv")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--scenarios", nargs="+", default=None)
    run.add_argument("--output-dir", default="outputs")
    judge = commands.add_parser("evaluate-judgments", help="Score completed adjudicated human labels")
    judge.add_argument("experiment_directory", type=Path)
    judge.add_argument("judgments", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download_imdb
        print(download_imdb(args.destination))
    elif args.command == "run":
        from .experiment import run_study
        options = {"imdb_path": args.imdb_path, "device": args.device, "output_dir": args.output_dir}
        if args.scenarios:
            options["scenarios"] = tuple(args.scenarios)
        print(run_study(profile(args.profile, **options)))
    else:
        from .annotations import evaluate_judgments
        result = evaluate_judgments(args.experiment_directory, args.judgments)
        print(f"Evaluated {result.query_id.nunique()} independently judged queries.")


if __name__ == "__main__":
    main()
