= Materials and Methods

The companion Zenodo repository @bib-zenodo contains the source code required to reproduce this manuscript.
Pandas @bib-pandas and Polars @bib-polars were used to process tabular data.
Matplotlib @bib-matplotlib was used for plotting figures.
GNU parallel @bib-parallel was used for managing parallel execution of some analyses.
This manuscript is written in and rendered using Typst @bib-madje-2022-typst @bib-haug-2022-typst.

== Software Environments

Software environments were managed with pixi @bib-pixi.
Packages were downloaded from the conda-forge @bib-conda-forge and Bioconda @bib-bioconda Conda channels.
The exact software versions used in this work are defined in configuration files within the manuscript's companion repository.

== Data preprocessing

All genome assemblies and annotations were downloaded from MaizeGDB @bib-NAM-genomes @bib-MaizeGDB.
Version 5 of the B73 assembly and annotation was used.
For PanAnd, version 1 of the assemblies and version 2 of the annotations were used.
B73 and other NAM parent RNA-seq data were downloaded from ArrayExpress accessions E-MTAB-8628 and E-MTAB-8633, respectively.
Other Andropogoneae RNA-seq data were downloaded from NCBI accession PRJNA1098707.
Transcript quantifications were obtained using quantify-RNA-pipeline @bib-quantify-RNA-pipeline.
RNA-seq samples with less than 5 million mapped reads were dropped from further analysis.
eggnog-mapper @bib-eggnog-mapper was used to assign proteins to Poales orthogroups.

_Zea mays_ genes were assigned to the test set.
90% of orthogroups were randomly chosen as training orthogroups, with the other 10% used for validation.
_Zea diploperennis_ and _Tripsacum zopliotense_ genes in the validation orthogroups were used as the validation set.
Genes in the training orthogroups in all remaining Andropogoneae genomes were assigned to the training set.

Annotations were processed using gffutils @bib-gffutils.
For each gene, the highest expressed transcript across all tissues was selected as a representative gene model.
TPM values from other transcripts of the same gene that share the same transcription start site were added to the chosen transcript's TPM.
For the purposes of computing max expression, only leaf, shoot, and floral tissues were used as only those tissues had sufficient sampling across all species.
The any tissue and leaf on/off expression classification task targets were binarized from the max expression and leaf absolute expression regression task targets (TPM).
Specifically, TPM values of zero were kept as zero (unexpressed) and TPM values greater than zero were set to one (expressed).

== Model architectures

Exact hyperparameter settings for each architecture is specified in configuration files within the companion repository.
DanQ @bib-DanQ and FNetCompression @bib-FNetCompression were both converted to PyTorch @bib-PyTorch, keeping all hyperparameters identical.
Miniformer is a scaled-down version of the Enformer @bib-Enformer architecture, with lower model dimensions and fewer layers.
HyenaDNA @bib-HyenaDNA was used in the "tiny" configuration.
Classification architectures were identical to their regression counterparts except that the final activation function was changed to sigmoid.

== Training

PyTorch Lightning @bib-PyTorch-Lightning and Hydra @bib-Hydra were used to orchestrate the training process and provide an interface for the data loader.
MLFlow @bib-MLFlow was used to track experiment parameters and metrics as well as store model artifacts.
As input, 1,026 base pairs upstream from the translation start site were extracted.
The sequence was reverse complemented if the transcript was on the negative strand.
If the model used 1-hot encoded sequence as input, the sequence was 1-hot encoded and then padded or trimmed as needed to be exactly 1,026 base pairs in length.
If the model used tokens as input, the input sequence was tokenized to a max length of 1,026, padding as needed.
If the task was regression (max or absolute leaf expression), TPM values were $attach(log, br: 10)("TPM" + 1)$ transformed.
Training continued until the validation loss failed to decrease after three epochs.
The model checkpoint at the end of the epoch with the lowest validation loss was kept.
Each combination of architecture and task was run three times with different initial seeds to estimate model robustness.
Any runs that failed to converge were restarted with a different seed value until a total of three converged runs were obtained.

== Ablation

"By genome" ablation was performed by filtering to one, two, three, four, or eight training genomes, in order of increasing phylogenetic distance from the test set.
The first eight genomes, in order, were _Tripsacum dactyloides_ "FL_9056069_6", "McKain_334-5", _Elionurus tripsacoides_, _Hemarthria compressa_, _Thelepogon elegans_, _Sorghastrum nutans_, _Ischaemum rugosum_, and _Pogonatherum paniceum_.
"Random" ablation randomly sampled an equivalent number of observations from the total set of observations from all 15 training species.
For example, if the "By genome" ablation had two genomes with 30,000 and 35,000 observations, then the corresponding "random" ablation experiment would randomly sample 65,000 observations from the total training set.
Each ablation run was repeated six times to measure robustness.

== Ortholog contrast

The first DanQ training run model was used to predict the expression for each transcript in the test set.
All possible pairs of orthologs within each orthogroup were generated for the contrast.
Orthogrups were filtered to those that contained between 20 and 35 members, to avoid private genes and retroelements and have sufficient sample sizes to calculate correlation and auROC.

== Saliency map

Captum @bib-captum was used to compute saliency.
For each position, the absolute value of saliency for each channel was summed.
The mean and standard deviation of this sum was computed across all B73 genes.
