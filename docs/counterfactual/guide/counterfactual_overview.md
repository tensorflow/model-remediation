# Counterfactual Logit Pairing for Model Remediation

Counterfactual Logit Pairing (CLP) is a technique within the TensorFlow Model
Remediation Library that seeks to ensure that a model’s prediction doesn’t
change when a sensitive attribute referenced in an example is either removed or
replaced. For example, in a toxicity classifier, examples such as "I am a man"
and "I am a lesbian" should not have a different prediction of toxicity.

For an in-depth discussion on this topic, see the research on [counterfactual
fairness](https://arxiv.org/abs/1703.06856), [adversarial logit pairing](https://arxiv.org/abs/1803.06373),
and [counterfactual logit pairing](https://arxiv.org/abs/1809.10610).

## When should you use Counterfactual Logit Pairing?

CLP addresses the scenario where a change in a sensitive attribute referenced
in a feature changes the prediction (when the prediction should not have
changed). In doing so, it attempts to answer the question: Is this model
susceptible to changing its prediction based solely on the presence of an
identity attribute? See the [research paper](https://arxiv.org/abs/1703.06856)
for details on counterfactual fairness.

This issue was seen in the
[Perspective API](https://www.perspectiveapi.com/#/home){: .external}, an ML
tool used by developers and publishers to analyze the content of comments for
potentially offensive or *toxic* text. The Perspective API takes comment text
as input and returns a score from 0 to 1 as an indication of the probability
that the comment is toxic. For example, a comment like “You are an idiot” may
receive a probability score of 0.8 for toxicity, indicating how likely it is
that a reader would perceive that comment as toxic.

After the initial launch of the Perspective API, external users discovered a
positive correlation between identity terms containing information on race or
sexual orientation and the predicted toxicity score. For example, the phrase "I
am a lesbian" received a score of 0.51, while “I am a man” received a lower
score of 0.2. In this case, the identity terms were not being used pejoratively,
so there should not be such a significant difference in the score. For more
information on the Perspective API, see the blog post on
[unintended bias and identity terms](https://medium.com/jigsaw/unintended-bias-and-names-of-frequently-targeted-groups-8e0b81f80a23){: .external}.

## How can I measure the effect of Counterfactual Logit Pairing?

If you have assessed your machine learning model and determined that changes in
predictions due to changes in specific sensitive attributes would be harmful,
then you should measure the prevalence of this issue. In the case of a
binary or multi-class classifier, a *flip* is defined as a classifier giving a
different decision (such as changing a prediction from toxic to not toxic)
when the sensitive attribute referenced in the example changes. When assessing
the prevalence of *flips*, you can look at *flip count* and *flip rate*. By
taking into account the potential user harm caused by a *flip* and the
frequency that flips occur, you can determine if this is an issue that should be
addressed by applying CLP. For more information about these metrics,
refer to the [Fairness Indicators guide](https://www.tensorflow.org//responsible_ai/fairness_indicators/guide/guidance#which_metrics_should_i_choose).

## On what model types can I apply Counterfactual Logit Pairing?

This technique can be used with binary and multi-class classifiers of
different types of data such as text, images, and videos.

## When is Counterfactual Logit Pairing not right for me?

CLP is not the right method for all situations. For example, it is not
relevant if the presence or absence of an identity term legitimately
changes the classifier prediction. This may be the case if the classifier
aims to determine whether the feature is referencing a particular identity
group. This method is also less impactful if the unintended correlation
between classifier result and identity group has no negative repercussions
on the user.

CLP is useful for testing whether a language model or toxicity classifier is
changing its output in an unfair way (for example classifying a piece of text
as toxic) simply because terms like “Black”, “gay”, “Muslim” are present in
the text. CLP is not intended for making predictions about individuals, for
example by manipulating the identity of an individual. See this [paper](https://arxiv.org/abs/2102.05085)
for a more detailed discussion.

It is important to keep in mind that CLP is one technique in the [Responsible
AI Toolkit](https://www.tensorflow.org/responsible_ai) that is specifically
designed to address the situation where sensitive attributes referenced in
features changes the prediction. Depending on your model and use case, it
may also be important to consider whether there are performance gaps for
historically marginalized groups, particularly as CLP may affect group
performance. This can be assessed with [Fairness Indicators](https://tensorflow.org/responsible_ai/fairness_indicators/guide)
and addressed by [MinDiff](https://tensorflow.org/responsible_ai/model_remediation/min_diff/guide/mindiff_overview)
that is also in the TensorFlow Model Remediation Library.

You should also consider whether your product is an appropriate use for
machine learning at all. If it is, your machine learning workflow should
be designed to known recommended practices such as having a well defined
model task and clear product needs.

## How does Counterfactual Logit Pairing work?

CLP adds a loss to the original model that is provided by logit pairing an
original and counterfactual example from a dataset. By calculating the
difference between the two values, you penalize the differences of the
sensitive terms that are causing your classifier prediction to change.
This work was based on research on [adversarial logit pairing](https://arxiv.org/abs/1803.06373)
and [counterfactual logit pairing](https://arxiv.org/abs/1809.10610).
