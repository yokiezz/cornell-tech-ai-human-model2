"""A small, training-only introduction to the competition texts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from model2.cli import digest
from model2.data import partition, read_data


def describe_training(frame):
    """Keep simple measurements interpretable; do not infer authorship rules."""
    measured = frame.assign(
        words=frame.text.str.split().str.len(),
        characters=frame.text.str.len(),
        newlines=frame.text.str.count("\n"),
    )
    rows = []
    for label, name in [(0, "Human"), (1, "AI")]:
        group = measured[measured.label == label]
        for feature in ["words", "characters", "newlines"]:
            values = group[feature]
            rows.append({"class": name, "feature": feature, "rows": len(group),
                         "mean": values.mean(), "min": values.min(),
                         "q25": values.quantile(.25), "median": values.median(),
                         "q75": values.quantile(.75), "max": values.max()})
    return measured, pd.DataFrame(rows)


def plot_overview(measured, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    colors = ["#52677c", "#008080"]
    counts = measured.label.value_counts().reindex([0, 1])
    axes[0].bar(["Human", "AI"], counts, color=colors)
    for index, count in enumerate(counts):
        axes[0].text(index, count, str(count), ha="center", va="bottom")
    axes[0].set(title="Class balance", ylabel="Training examples", ylim=(0, counts.max() * 1.15))
    for ax, feature, title in zip(axes[1:], ["words", "newlines"],
                                  ["Whitespace-separated words", "Newline characters"]):
        # Shared bin edges make the two class distributions comparable.
        _, edges = pd.cut(measured[feature], bins=20, retbins=True)
        for label, name, color in [(0, "Human", colors[0]), (1, "AI", colors[1])]:
            ax.hist(measured.loc[measured.label == label, feature], bins=edges,
                    histtype="step", linewidth=1.8, color=color, label=name)
        ax.set(title=title, xlabel="Count per text", ylabel="Training examples")
        ax.legend(frameon=False)
    fig.suptitle(f"Explore the training partition only — {len(measured):,} examples")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--split", type=Path, default=Path("records/cornell/shared_split/split.csv"))
    parser.add_argument("--out", type=Path, required=True, help="New local output directory")
    args = parser.parse_args()

    # Load the original data, then select training IDs before describing labels/texts.
    raw = read_data(args.data, labeled=True, expected_rows=10000)
    split = pd.read_csv(args.split, dtype=str, keep_default_na=False)
    training, _ = partition(raw, split)
    measured, summary = describe_training(training)
    examples = pd.concat([measured[measured.label == label].sample(n=3, random_state=42)
                          for label in [0, 1]], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    # Raw examples remain local, just like the original competition data.
    examples.to_csv(args.out / "examples_local_only.csv", index=False)
    plot_overview(measured, args.out / "overview.png")
    manifest = {"training_rows": len(training), "example_seed": 42,
                "examples_per_class": 3, "data_sha256": digest(args.data),
                "split_sha256": digest(args.split), "source_sha256": digest(__file__),
                "scope": "Training partition only; no validation/test descriptions or model fitting"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "# Training-data exploration\n\n"
        "![Class balance, text length and formatting](overview.png)\n\n"
        "Open `summary.csv` for counts and distribution summaries, and "
        "`examples_local_only.csv` for three seeded examples per class. "
        "Labels are supplied annotations, not conclusions from reading the prose.\n\n"
        "Ask: Are the classes balanced? How much do length distributions overlap? "
        "Is a difference typical or driven by the tails? Do the examples show "
        "variation within each class? Six examples cannot establish a genre or style rule.\n\n"
        "Words are whitespace-separated tokens; characters include whitespace; "
        "newlines count literal newline characters, not rendered lines. "
        "These are descriptive measurements, not classifier features or causal explanations. "
        "No cleaning, feature selection, or model changes are applied. "
        "Keep raw example exports out of Git.\n"
    )
    print(summary.to_string(index=False))
    print(f"\nOpen {args.out / 'README.md'}")


if __name__ == "__main__":
    main()
