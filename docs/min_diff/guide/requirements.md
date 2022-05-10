# MinDiff Requirements

## When should I use MinDiff?
Apply MinDiff in instances where your model performs well
generally, but produces harmful errors more frequently on examples belonging to
a sensitive group, and you wish to close the performance gap. The sensitive
groups of interest may vary depending on your use case, but often include
protected classes, such as race, religion, gender, sexual orientation, and more.
Throughout this document, we will use “sensitive group” to refer to any set of
examples belonging to a protected class.

There are two primary conditions for using MinDiff to address underperforming
slices of data:


-   You have already tuned and evaluated your model, identifying metrics that
show underperforming slices of data. This must be done *before* applying model
remediation.
-   You have, or can obtain, a sufficient number of relevant labeled examples
belonging to the underperforming group (more details below).

Note: If you’re new to evaluating for fairness, we recommend reviewing this
module in Google’s Machine Learning Crash Course, which provides an introduction
to this topic, and this guide for a broader discussion on the topic.

MinDiff is one of many techniques for remediating unequal behavior. In
particular, it may be a good choice when you’re trying to directly equalize
performance between groups. MinDiff can be used in conjunction with other
approaches, such as data augmentation and others, which may lead to better
results. However, if you need to prioritize which technique to invest in, you
should do so according to your product needs.

When applying MinDiff, you may see performance degrade or shift slightly for
your best performing groups, as your underperforming groups improve. This
tradeoff is expected, and should be evaluated in the context of your product
requirements. In practice, we have often seen that MinDiff does not cause top
performing slices to drop below acceptable levels, but this is application-
specific and a decision that needs to be made by the product owner.

### On what model types can I apply MinDiff?
MinDiff has been shown to be consistently effective when applied to *binary
classifiers.* Adapting the method for other applications is possible, but has
not been fully tested. Some work has been done to show success in multi-
classification and ranking tasks<sup>1</sup> but any use of MinDiff on these or
other types of models should be considered experimental.

### On what metrics can I apply MinDiff?
MinDiff may be a good solution when the metric you’re trying to equalize across
groups is *false positive rate (FPR)*, or *false negative rate (FNR)*, but it may
work for other metrics. As a general rule, MinDiff may work when the metric
you’re targeting is a result of differences in the score distributions between
examples belonging to a sensitive group and examples not belonging to a
sensitive group.

## Building your MinDiff dataset
When preparing to train with MinDiff, you’ll need to prepare three separate
datasets. As with regular training, your MinDiff datasets should be
representative of the users your model serves. MinDiff may work without this but
you should use extra caution in such cases.

Assuming you’re trying to improve your model’s FPR for examples belonging to a
sensitive class, you’ll need:
1. The original training set - The original dataset that was used for training
your baseline model
2. The MinDiff sensitive set - A dataset of examples belonging to the sensitive
class with *only* negative ground truth labels. These examples will be used only
for calculating the MinDiff loss.
3. The MinDiff non-sensitive set - A dataset of examples *not* belonging to the
sensitive class with *only* negative ground truth labels. These examples will be
used only for calculating the MinDiff loss.

Note: The examples in your MinDiff datasets may overlap partially, or completely
with your original training dataset.

When using the library, you will combine all three of these datasets into a
single dataset, which will serve as your new training set.

### Picking examples for MinDiff
It may have seemed counterintuitive in the example above to carve out sets of
*negatively* labeled examples if you are primarily concerned with disparities in
*false positive rate*. However, remember that a false positive prediction comes
from a negatively labeled example incorrectly classified as positive.

When collecting your data for MinDiff, you should pick examples where the
disparity in performance is evident. In our example above, this meant choosing
negatively labeled examples to address FPR. Had we been interested in targeting
FNR, we would have needed to choose positively labeled examples.

### How much data do I need?
Good question--it depends on your use case! Based on your model architecture,
data distribution, and MinDiff configuration, the amount of data needed can vary
significantly. In past applications, we have seen MinDiff work well with 5,000
examples in each MinDiff training set (sets 2 and 3 in the previous section).
With less data, there is increased risk of lowered performance, but this may be
minimal or acceptable within the bounds of your production constraints. After
applying MinDiff, you will need to evaluate your results thoroughly to ensure
acceptable performance. If they are unreliable, or do not meet performance
expectations, you may still want to consider gathering more data.

## When is MinDiff *not* right for me?
MinDiff is a powerful technique that can provide impressive results, but this
does not mean that it is the right method for all situations. Applying it
haphazardly does not guarantee that you will achieve an adequate solution.

Beyond the requirements discussed above, there are cases where MinDiff may be
technically feasible, but not suitable. You should always design your ML
workflow according to known recommended practices. For instance, if your model
task is ill-defined, the product needs unclear, or your example labels overly
skewed, you should prioritize addressing these issues. Similarly, if you do not
have a clear definition of the sensitive group, or are unable to reliably
determine whether examples belong to the sensitive group, you will not be able
to apply MinDiff effectively.

At a higher level, you should always consider whether your product is an
appropriate use for ML at all. If it is, consider the potential vectors for user
harm it creates. The pursuit of responsible ML is a multi-faceted effort which
aims to anticipate a broad range of potential harms; MinDiff can help mitigate
some of these, but all outcomes deserve careful consideration.

<sup>1</sup>Beutel A., Chen, J., Doshi, T., Qian, H., Wei, L., Wu, Y., Heldt,
L., Zhao, Z., Hong, L., Chi, E., Goodrow, C. (2019). [Fairness in Recommendation
Ranking through Pairwise
Comparisons.](https://dl.acm.org/doi/abs/10.1145/3292500.3330745)
