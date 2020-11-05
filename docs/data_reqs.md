# Min Diff: Data Requirements

## When should I use min diff?
There are three primary conditions for using min diff to improve your model’s
performance for underperforming slices of data:

-   The model is a *classifier*
-   The target metric you’re trying to improve is False Positive Rate (FPR) or False Negative Rate (FNR)
-   You have a sufficient number of examples belonging to the underperforming (or can obtain enough examples)

Further, you might consider MinDiff when performing interventions on your
dataset is impractical, either because it’s too difficult to collect more data,
or because the data already represents the real-world distribution, but the
real-world distribution is skewed.

## Building your min diff set
When preparing to train with MinDiff, you’ll need to prepare four slices of your
dataset. Assuming you’re trying to improve your model’s FPR for examples
belonging to a sensitive class, you’ll need:

1.   A main training set
2.   A min diff training set made up of only ground truth negative examples belonging to the sensitive class
3.   A min diff training set made up of only ground truth negative examples *not* belonging to the sensitive class
4.   An evaluation set, sliced by membership to the sensitive class

Note: It may seem counterintuitive to carve out sets of ground truth *negative*
examples if you’re primarily concerned with disparities in *false positive
rate*, but remember that a false positive prediction is a ground truth negative
example incorrectly classified as positive.

## How much data do I need?

While there isn’t strict guidance on the minimum size of each slice of data, we
generally recommend trying min diff when one has no fewer than 5,000 examples in
any set. Still, even a set in that range may be insufficient, and this may still
limit performance improvements.
