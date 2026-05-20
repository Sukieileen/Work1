import argparse
import csv
import os


FIELDNAMES = [
    "train_direction",
    "zero_target",
    "method",
    "precision",
    "recall",
    "f1",
    "threshold",
    "threshold_source",
    "checkpoint",
    "total",
    "normal",
    "anomalous",
    "tp",
    "tn",
    "fp",
    "fn",
    "known_event_count",
    "zero_event_count",
    "zero_instance_count",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--known_direction", required=True)
    parser.add_argument("--zero_target", default="SPIRIT")
    parser.add_argument("--method", default="MetaLog")
    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as reader:
        rows = list(csv.DictReader(reader))
    matches = [
        row for row in rows
        if row.get("train_direction") == args.known_direction
        and row.get("zero_target") == args.zero_target
        and row.get("method") == args.method
    ]
    if not matches:
        raise SystemExit("No matching row found in %s" % args.source)

    row = matches[-1]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=FIELDNAMES)
        csv_writer.writeheader()
        csv_writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


if __name__ == "__main__":
    main()
