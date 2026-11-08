# A first look at the training data

Purpose: understand the examples before discussing model choices. This short module
is intended to be read and run by a project member, even when an agent can produce
the summaries automatically. It was added after the November 7 pilot; it does not
claim that these exploration steps historically preceded the recorded experiments.

## Run and inspect

From the repository root, with project dependencies and the `plots` extra available:

```bash
PYTHONPATH=src python -m model2.explore --out artifacts/cornell/exploration-001
```

Use a new output suffix for each rerun. The module uses the confirmed ID split and
describes only its 8,000 training examples. It does not use the supplied starter,
validation labels, unlabeled data, or test data for exploration. It does not train
a model or change the inputs. The separate `model2.cornell audit` command handles
dataset integrity, starter membership, and overlap checks.

1. **Load and check scope.** Read the manifest and confirm which data and split were
   used. Human = 0; AI = 1. The class-count chart shows how many examples of each
   class the training partition contains.
2. **Compare distributions.** Open `overview.png` and `summary.csv`. Compare word
   counts and newline counts across classes. Look at overlap and quartiles, not
   just means. Words are whitespace-separated tokens, not linguistic word counts;
   newline characters are not the number of lines displayed in an editor.
3. **Read examples.** Open `examples_local_only.csv` in a CSV viewer that supports
   multiline cells. Three examples per class are sampled with seed 42. Describe
   apparent genre, tone, and formatting, while keeping these impressions separate
   from verified source/generator information, which the data do not provide.
4. **Write a short interpretation.** Record one observation, one possible
   explanation, and one check that could distinguish that explanation from an
   alternative. For example, a newline difference might reflect formatting or
   source composition; it does not establish an authorship rule. Any proposed
   preprocessing ablation belongs in a separately specified training-only experiment.

The six examples are a small reading sample, not an error analysis or proof of a
class-wide pattern. Review more training examples before generalizing. Do not
change labels based on how a passage sounds. Keep raw excerpts and generated
outputs local; promote only deliberately reviewed aggregate findings into records.

Implementation: [`src/model2/explore.py`](../src/model2/explore.py).
