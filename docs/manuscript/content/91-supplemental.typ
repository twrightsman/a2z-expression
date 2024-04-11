#pagebreak(weak: true)
= Supplemental Material

#set figure(
  kind: "figure-supplemental",
  supplement: [Supplemental Figure]
)

#figure(
  image(
    "figures/performance_across_prsn.svg",
    width: 80%
  ),
  caption: [
    Pearson correlation in the regression tasks across all genes and data splits.
    Each plot shows the performance of all architectures on a single task.
    Error bars represent one standard deviation from the mean in each direction.
  ],
) <fig_performance_across_prsn>

#figure(
  image(
    "figures/DanQ_pred.png",
    width: 75%
  ),
  caption: [
    DanQ predictions on the test set across all tasks.
    The model from the first training run was used for predictions.
    Regression tasks (top) are on the $log$ scale.
    Color in the regression task histogram scatterplots represents the number of observations within that area.
  ],
) <fig_DanQ_pred>

#figure(
  image(
    "figures/performance_within_prsn.svg",
    width: 80%
  ),
  caption: [
    Distributions of Pearson correlation within orthogroups for each task.
    Architectures are sorted from highest (left) average within-orthogroup Pearson correlation to lowest (right).
    Bars within the violins represent the mean of the distribution.
  ],
) <fig_performance_within_prsn>
