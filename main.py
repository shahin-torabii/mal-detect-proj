"""Single entry point for the whole pipeline.

Usage:
    python main.py preprocess          # build TF-IDF cache (optional branch)
    python main.py pretrain            # stage 1: SupCon encoder pretraining
    python main.py train-classifier    # stage 2: train classifier head
    python main.py evaluate            # stage 3: evaluate on test set

All stages read settings from config.yaml (pass --config to use another file).
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Malware detection pipeline")
    parser.add_argument(
        "stage",
        choices=["preprocess", "pretrain", "train-classifier", "evaluate"],
        help="Which pipeline stage to run",
    )
    parser.add_argument("--config", default="config.yaml")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--config", args.config, *remaining]

    if args.stage == "preprocess":
        import src.preprocessing.tfidf as mod
    elif args.stage == "pretrain":
        import src.training.train_supcon as mod
    elif args.stage == "train-classifier":
        import src.training.train_classifier as mod
    else:
        import src.evaluation.evaluate as mod

    mod.main()


if __name__ == "__main__":
    main()
