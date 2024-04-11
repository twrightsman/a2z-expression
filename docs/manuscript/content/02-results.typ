= Results

#figure(
  image(
    "figures/methods.svg",
    width: 80%
  ),
  caption: [
    Methods overview:
    a) data splits;
    b) models, features, and targets;
    c) orthogroup-guided splitting;
    d) metrics (across- vs. within-gene performance)
  ],
) <fig_methods>

== Current genomic deep learning architectures generalize across species

We trained all four architectures on genomic sequence and RNA-seq data from 15 species within the Andropogoneae clade (@fig_methods#text[a]).
Our validation set consisted of the two sampled species closest to _Zea mays_, _Tripsacum zopilotense_ and _Zea diploperennis_, all three of which fall within the Tripsacinae subtribe that diverged a few (0.6--4) million years ago @bib-andropogoneae-divergence @bib-portrait-genus.
Our test set was the 26 inbred parents of the maize NAM population @bib-NAM @bib-NAM-genomes, held out until hyperparameters were frozen.
As input, we extracted 1,026 base pairs upstream of the translation start site to match HyenaDNA's "tiny" configuration (@fig_methods#text[b]).
We trained and evaluated all architectures on two regression tasks, maximum expression across tissues and absolute expression in leaf, as well as two classification tasks, expressed in any tissue and expressed in leaf (@fig_methods#text[b]).

#figure(
  image(
    "figures/performance_across.svg",
    width: 60%
  ),
  caption: [
    a--d)
    Model performance across all genes and data splits.
    Each subfigure shows the performance of all architectures on a single task.
    Error bars represent one standard deviation from the mean in each direction.
  ],
) <fig_performance_across>

Though benchmarking of sequence-based models has been done within humans @bib-benchmark-human and across species in the training set @bib-Basenji @bib-FloraBERT, there has been little evaluation on entirely held out species.
To establish a baseline in plants, we measured performance of all architectures, tasks, genes, and data splits (@fig_performance_across#text[a--d]).
Rankings by Spearman correlation on the test set are inconsistent, except that DanQ performed the best or tied closely with the best across all tasks.
Remarkably, DanQ performs only slightly worse (@fig_performance_across#text[a;] $Delta r = 0.09$) than Enformer in a recent within-human single tissue benchmark @bib-personal-transcriptome-variation despite predicting on an unobserved species.
Despite having moderate Spearman and Pearson correlation (@fig_performance_across_prsn), DanQ's predictions on the test set are still underwhelming (@fig_DanQ_pred).
We observed test set auROC scores in the any expression task slightly lower than previous results @bib-hai-jacob-cnn on promoter expression classification models trained and tested only within maize.
Taken together, these results support modern genomic deep learning architectures are capable of generalizing almost as well across closely related species as they do within species.

== Data quantity matters more than composition for modeling RNA abundance across species

#figure(
  image(
    "figures/performance_within_ablation.svg",
    width: 95%
  ),
  caption: [
    a)
    Validation set performance of DanQ on the maximum expression task across varying training set sizes and compositions.
    Points on the lines are mean Pearson correlation across replicate training runs.
    The standard deviation across replicates is shaded around the mean line.
    The exponent scale of the bottom axis is denoted in bold on the right.
    b--e)
    Distributions of model performance within orthogroups for each task.
    Architectures are sorted from highest (left) average within-orthogroup performance to lowest (right).
    Bars within the violins represent the mean of the distribution.
  ],
) <fig_performance_within_ablation>

Despite the growing number of plant genomes with transcriptomic data @bib-plant-gene-atlas, each genome added to the training set increases training time and may give diminishing returns.
We measured changes in DanQ's performance on progressively larger fractions of the training data, iteratively adding sequences from a set of genomes or randomly from across all training genomes.
Pearson correlation on the validation set rises until approximately 200,000 observations when it begins to show diminishing returns for larger training set sizes (@fig_performance_within_ablation#text[a]).
However, the slope remains positive between the half size and full size data points, suggesting room for improvement with further observations.
Comparing iteratively adding whole genomes to randomly sampling an equivalent number of alleles from the entire training set, there are only substantial differences when using less than 8 genomes, with random performing worse than 4 whole genomes.
The ablation results clearly support the use of further data to achieve higher performance across genes, which can come from sequencing additional related species.

== Current architectures poorly generalize across individuals of an inbred maize panel

Recent work @bib-personal-transcriptome-variation has shown that current models poorly explain expression variation across individuals.
Since our test set is a collection of maize alleles with an order of magnitude more diversity than humans @bib-maize-HapMap2, we looked at the distribution of test set performance within each orthogroup and expected to see similarly low or even lower performance.
We only considered orthogroups that had at least 20 orthologs to have sufficient sample size for calculating correlation or auROC.
We saw much lower average within-orthogroup Spearman and Pearson correlations as well as auROC compared to the global across-gene metrics, except for the any expression task (@fig_performance_within_ablation#text[b--e;] @fig_performance_within_prsn), which also shows clear differences between architectures.
The average within-orthogroup Spearman correlation in the single tissue regression task is double ($r = 0.092$) than what was observed with Enformer ($r = 0.045$) in humans.

#figure(
  image(
    "figures/interpretation.png",
    width: 95%
  ),
  caption: [
    Left: Predicted versus observed $attach(log, br: 10)$ expression change in leaf between all NAM ortholog promoter allele pairs within an orthogroup.
    Percentages in the middle of each quadrant display the proportion of non-zero data points in that quadrant.
    Right: Saliency map for DanQ trained on maximum expression task.
    The mean across all B73 genes is plotted as a line, with a single standard deviation shaded above and below.
  ],
) <fig_interpretation>

As an alternative allele-level comparison, we also looked at how well DanQ predicted expression differences between all pairs of maize ortholog promoter alleles within a orthogroup.
We observed a general positive relationship between the two, but there is still quite a bit of noise in the predictions (@fig_interpretation, left).
The Pearson and Spearman correlation coefficients between the observed and predicted fold changes were only 0.22 and 0.08, respectively.
Strikingly, pairs of orthogroups that are two orders of magnitude apart in expression level are still sometimes predicted with the incorrect direction.
Despite current architectures showing promising across gene performance in unobserved species, they still struggle across shorter evolutionary timescales, similar to what was seen in humans @bib-personal-transcriptome-variation.

== The maximum expression regression model focuses on the core promoter region

Based on theory and prior interpretation work on expression models @bib-AgroNT, we hypothesized our expression models would also pay most attention to the region surrounding the transcription start site.
Looking at the average saliency map for DanQ across all B73 genes on the maximum expression task we see that DanQ indeed focuses on the core promoter region and the 5' UTR (@fig_interpretation, right).
There is relatively high variance in the gradient around the transcription start site, taping off with increasing distance, though decaying slower in the UTR than promoter.
This hyperfocus on the core promoter region could be why DanQ and other architectures struggle to distinguish expression differences in highly related sequences, since functional mutations are less likely to accumulate in this highly constrained region.
