"""Copy measured artifacts into the submission folder without large checkpoints."""
import argparse
import shutil
from pathlib import Path


def package(output, submission):
    output, submission = Path(output), Path(submission)
    for marker in ("evidence/completed.json", "evidence/B06_supplement_completed.json"):
        if not (output / marker).is_file():
            raise FileNotFoundError(output / marker)
    for name in ("results.xlsx", "report.md"):
        shutil.copy2(output / name, submission / name)
    for name in ("curves", "predictions", "evidence"):
        shutil.copytree(output / name, submission / name, dirs_exist_ok=True)
    print(f"Submission artifacts copied to {submission}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    package(args.output, args.submission)
