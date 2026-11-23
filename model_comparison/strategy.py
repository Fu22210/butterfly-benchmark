"""Select confidence and corroboration rules on team data, or apply a frozen rule."""
import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluate import iou, load, score, unique_keys


THRESHOLDS = {
    "dino": (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    "owl": (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
}
N_FOLDS = 5
SEED = 20261120


@dataclass(frozen=True)
class Rule:
    detector: str
    confidence: float | None = None
    confirm_iou: float | None = None

    def __post_init__(self):
        if self.detector == "qwen" and self.confidence is None and self.confirm_iou is None:
            return
        if self.detector not in THRESHOLDS or self.confidence is None:
            raise ValueError("Unknown detector configuration")
        if not THRESHOLDS[self.detector][0] <= self.confidence <= 1:
            raise ValueError("Requested confidence falls outside the recorded range")
        if self.confirm_iou is not None and not 0 < self.confirm_iou <= 1:
            raise ValueError("Invalid confirmation IoU")

    @property
    def models(self):
        return (self.detector,) if self.confirm_iou is None else (self.detector, "qwen")


def load_inputs(directory, models, *, timings=False):
    boxes = {}
    confidence = {}
    seconds = {}
    for model in models:
        boxes[model] = load(directory / f"{model}.jsonl")
        if model == "qwen" and not timings:
            continue
        journal = directory / f"{model}.journal.jsonl"
        rows = {}
        with journal.open() as stream:
            for line in stream:
                row = json.loads(line, object_pairs_hook=unique_keys)
                if row.get("error") is None:
                    rows[row["photo_id"]] = row
        if rows.keys() != boxes[model].keys():
            raise ValueError(f"{model}: prediction and journal IDs differ")
        seconds[model] = sum(row["seconds"] for row in rows.values())
        if model == "qwen":
            continue
        confidence[model] = {}
        for i, row in rows.items():
            encoded = [(b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"])
                       for b in row["boxes"]]
            values = row["scores"]
            if encoded != boxes[model][i] or len(encoded) != len(values):
                raise ValueError(f"{model} photo {i}: scores do not match the stored boxes")
            if any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1 for s in values):
                raise ValueError(f"{model} photo {i}: invalid score")
            confidence[model][i] = values
    first = next(iter(boxes.values()))
    if any(rows.keys() != first.keys() for rows in boxes.values()):
        raise ValueError("The input models processed different photo IDs")
    return boxes, confidence, seconds


def predict(rule, boxes, confidence):
    if rule.detector == "qwen":
        return boxes["qwen"]
    result = {}
    for i, candidates in boxes[rule.detector].items():
        strong = [b for b, s in zip(candidates, confidence[rule.detector][i]) if s >= rule.confidence]
        if rule.confirm_iou is None:
            result[i] = strong
            continue
        proposals = boxes["qwen"][i]
        pairs = [(iou(q, d), qi, di) for qi, q in enumerate(proposals) for di, d in enumerate(strong)]
        confirmed, used = set(), set()
        for overlap, qi, di in sorted(pairs, key=lambda item: -item[0]):
            if overlap < rule.confirm_iou:
                break
            if qi not in confirmed and di not in used:
                confirmed.add(qi)
                used.add(di)
        result[i] = [q for qi, q in enumerate(proposals) if qi in confirmed]
    return result


def write_predictions(path, boxes):
    lines = []
    for i, rects in sorted(boxes.items()):
        norm = [dict(left=x, top=y, width=r-x, height=b-y) for x, y, r, b in rects]
        lines.append(json.dumps(dict(photo_id=i, boxes=norm, error=None), allow_nan=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(gold, predictions):
    return {str(t): score(gold, predictions, sorted(gold), t) for t in (0.5, 0.75)}


def tune(args):
    gold = load(args.gold)
    boxes, confidence, seconds = load_inputs(args.inputs, ("dino", "owl", "qwen"), timings=True)
    if len(gold) != 400 or boxes["qwen"].keys() != gold.keys():
        raise ValueError("Expected the same 400 team photos in every input")
    rules = [Rule("qwen")]
    for model, cutoffs in THRESHOLDS.items():
        for cutoff in cutoffs:
            rules.extend(Rule(model, cutoff, overlap) for overlap in (None, 0.30, 0.50))
    predictions = {rule: predict(rule, boxes, confidence) for rule in rules}

    def best(ids):
        def rank(rule):
            f1 = score(gold, predictions[rule], ids, 0.5)["f1"]
            return (f1, -sum(seconds[m] for m in rule.models))
        return max(rules, key=rank)

    ids = sorted(gold)
    random.Random(SEED).shuffle(ids)
    held_out = {}
    folds = []
    for fold in range(N_FOLDS):
        test = ids[fold::N_FOLDS]
        train = sorted(gold.keys() - set(test))
        rule = best(train)
        held_out.update({i: predictions[rule][i] for i in test})
        folds.append(dict(rule=asdict(rule), held_out_ids=test,
                          metrics=score(gold, predictions[rule], test, 0.5)))
    assert held_out.keys() == gold.keys()
    winner = best(sorted(gold))
    summary = [dict(rule=asdict(r), metrics=summarize(gold, p),
                    inference_seconds=sum(seconds[m] for m in r.models)) for r, p in predictions.items()]
    report = dict(seed=SEED, folds=folds, out_of_fold=summarize(gold, held_out),
                  selected=asdict(winner), full=summarize(gold, predictions[winner]), candidates=summary)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(asdict(winner), indent=2) + "\n")
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_predictions(args.out / "team_predictions.jsonl", predictions[winner])
    write_predictions(args.out / "out_of_fold.jsonl", held_out)
    print(json.dumps({k: report[k] for k in ("selected", "full", "out_of_fold")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("tune", help="Compare candidates using team gold only")
    fit.add_argument("gold", type=Path)
    fit.add_argument("inputs", type=Path)
    fit.add_argument("out", type=Path)
    apply = sub.add_parser("apply", help="Run a frozen rule on new model outputs")
    apply.add_argument("config", type=Path)
    apply.add_argument("inputs", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "tune":
        tune(args)
        return
    rule = Rule(**json.loads(args.config.read_text()))
    boxes, confidence, _ = load_inputs(args.inputs, rule.models)
    write_predictions(args.output, predict(rule, boxes, confidence))


if __name__ == "__main__":
    main()
