Open-Set Domain Adaptation for Gas Sensor Array Drift and Bearing Fault Diagnosis
====================================================================================

OVERVIEW
--------

This project provides an experimental pipeline for open-set domain adaptation (OSDA) under distribution shift. The framework is evaluated in two sensor-based applications:

1. Gas recognition under temporal sensor drift using the UCI Gas Sensor Array Drift Dataset.
2. Bearing fault diagnosis under changing operating conditions using the Paderborn University Bearing Dataset.

The main objective is to investigate whether domain-adversarial representation learning can be improved by explicitly structuring the latent feature space of known classes. The framework extends a standard Domain-Adversarial Neural Network (DANN) with class-aware compactness and margin-based separation.

Three model variants are considered:

- Standard DANN
- Class-Aware DANN
- Margin-Aware DANN

Unknown samples are evaluated using three alternative open-set scoring strategies:

- Maximum Softmax Probability (MSP)
- Energy-based scoring
- Prototype distance

Open-set detection performance is evaluated using AUROC, AUPR, and FPR95. The project also includes SHAP-based explainability analysis to investigate which input features contribute to prototype-distance-based unknown detection.


1. DATASETS
===========

1.1 Paderborn University Bearing Dataset
-----------------------------------------

Source
~~~~~~

The bearing data are obtained from the Bearing DataCenter of the Chair of Design and Drive Technology (KAt), Paderborn University, Germany.

Official dataset page:
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter

Dataset Description
~~~~~~~~~~~~~~~~~~~

The Paderborn Bearing DataCenter provides experimental bearing data for condition monitoring based on vibration and motor-current signals. The test rig was operated under different operating conditions to support evaluation of condition-monitoring methods under varying conditions.

For this project, bearing states are grouped into three diagnostic classes:

- Healthy
- Inner: inner-race damage
- Outer: outer-race damage

The experiments use vibration measurements from selected bearings and operating conditions.

Signal Preprocessing
~~~~~~~~~~~~~~~~~~~~

Raw vibration signals stored in .mat files are divided into fixed-length windows.

The preprocessing configuration used in this project includes:

- Window length: 4096 samples
- Overlap: 50%
- Step size: 2048 samples
- Sampling frequency: 64 kHz
- Final model input: 22 engineered time- and frequency-domain features

The extracted feature set includes measures derived from quantities such as:

- Mean
- Standard deviation
- Variance
- Root Mean Square (RMS)
- Peak amplitude
- Skewness
- Kurtosis
- Crest factor
- Impulse factor
- Shape factor
- Clearance factor
- Spectral centroid
- Spectral RMS
- Dominant frequency
- Spectral entropy
- Frequency-band energies

Features are standardized using StandardScaler fitted on the designated source-domain training data.

Open-Set Domain Adaptation Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operating condition is treated as the domain, while bearing condition/fault type is treated as the class.

The source operating condition is:

- N15_M07_F10

The target operating conditions are:

- N09_M07_F10
- N15_M01_F10
- N15_M07_F04

For each source-to-target domain shift, one of the three classes (Healthy, Inner, or Outer) is designated as the unknown class, while the remaining two classes constitute the known-class set.

This produces nine open-set domain adaptation conditions:

3 target operating conditions x 3 held-out classes = 9 conditions (P01-P09)

Source and target subsets are separated using measurement/file identifiers according to the experimental protocol implemented in the notebook.


1.2 UCI Gas Sensor Array Drift Dataset
---------------------------------------

Source
~~~~~~

The gas-sensor experiments use the Gas Sensor Array Drift Dataset from the UCI Machine Learning Repository.

Official dataset page:
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

Dataset DOI:
https://doi.org/10.24432/C5RP6W

Dataset Description
~~~~~~~~~~~~~~~~~~~

The dataset contains 13,910 measurements from an array of 16 chemical sensors used for discrimination among six gases. The measurements were collected over time and are organized into 10 batches, providing a real-world setting for investigating sensor drift.

The six gas classes are:

- Ammonia
- Acetaldehyde
- Acetone
- Ethylene
- Ethanol
- Toluene

Each observation contains 128 sensor features.

Preprocessing
~~~~~~~~~~~~~

The original .dat files are parsed into structured tabular data. The preprocessing pipeline:

1. Reads the gas-class label for each observation.
2. Extracts the 128 sensor features.
3. Maps class identifiers to descriptive gas names.
4. Associates observations with their temporal batches.
5. Standardizes the input features using source-domain statistics.

The scaler is fitted on source-domain known-class samples and then applied to the corresponding target and test data.

Leave-One-Gas-Out Open-Set Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Leave-One-Gas-Out (LOGO) protocol is used.

For each experiment:

- One gas is excluded from known-class training and treated as unknown.
- The remaining five gases constitute the known-class set.

The temporal structure used in this project is:

- Source domain: B1-B5
- Target adaptation/calibration domain: B6
- Test domains: B7-B10

Each of the six gases is independently held out as unknown. Evaluation across six held-out gases and four temporal test batches therefore produces:

6 unknown gases x 4 test batches = 24 matched open-set evaluation conditions.


2. METHODOLOGY
==============

2.1 Domain-Adversarial Neural Network
--------------------------------------

The core architecture is based on the Domain-Adversarial Neural Network (DANN) introduced by Ganin et al. (2016).

The network contains three main components:

1. Feature Extractor
2. Known-Class Classifier
3. Domain Classifier

A Gradient Reversal Layer (GRL) connects the shared representation to the domain classifier. During adversarial training, the feature extractor is optimized to retain information useful for known-class discrimination while reducing information that allows the domain classifier to distinguish source from target samples.


2.2 Standard DANN
-----------------

The Standard DANN objective combines:

- Known-class classification loss on labeled source samples
- Domain-adversarial loss for source and target samples

The Gradient Reversal Layer reverses the gradient from the domain-classification objective before it reaches the feature extractor.


2.3 Class-Aware DANN
--------------------

The Class-Aware DANN extends the standard DANN objective with a compactness loss.

The compactness term encourages latent representations belonging to the same known class to remain close to their corresponding class prototype/centroid. Its purpose is to preserve class structure while domain alignment is performed.


2.4 Margin-Aware DANN
---------------------

The Margin-Aware DANN further extends the Class-Aware model with an inter-class margin term.

This objective encourages separation between known-class prototypes in the learned latent space.

The complete representation-learning strategy therefore combines:

- Domain alignment
- Within-class compactness
- Between-class separation

The resulting representation is subsequently used for prototype-based unknown detection.


3. OPEN-SET UNKNOWN DETECTION
=============================

After model training, test samples are processed by the learned feature extractor and known-class classifier. Three open-set scoring approaches are evaluated.


3.1 Maximum Softmax Probability (MSP)
--------------------------------------

Maximum Softmax Probability uses the largest predicted softmax probability among the known classes.

For evaluation, an unknownness-oriented score can be expressed as:

Unknown Score = 1 - max_k p(y = k | x)

Higher values therefore indicate lower classifier confidence and greater evidence that a sample may be unknown.


3.2 Energy-Based Score
----------------------

Energy-based detection uses the classifier logits and a log-sum-exp formulation rather than relying only on the maximum softmax probability.

The implementation maintains a consistent score orientation during evaluation so that the resulting score can be interpreted as an unknownness measure when computing open-set metrics.


3.3 Prototype Distance
----------------------

Known-class prototypes are calculated in the learned latent representation space.

For a test sample, the prototype-distance score is the minimum Euclidean distance between its latent representation and the known-class prototypes:

Prototype Score = min_k || z(x) - mu_k ||_2

where:

- z(x) is the latent representation of sample x.
- mu_k is the prototype of known class k.

A larger minimum prototype distance indicates that the sample lies farther from all known-class prototypes and therefore provides stronger evidence for an unknown sample.


4. THRESHOLD CALIBRATION
========================

Thresholds for converting continuous unknownness scores into known/unknown decisions are calibrated without using unknown-class labels.

For the gas-sensor experiments, the calibration procedure uses known samples from the target adaptation domain (B6). Percentile-based thresholds are derived according to the orientation of the corresponding open-set score.

Test batches B7-B10 are kept separate from threshold calibration.


5. EVALUATION METRICS
=====================

Open-set detection performance is evaluated using:

AUROC
-----

Area Under the Receiver Operating Characteristic Curve.

AUROC measures the ability of an unknownness score to discriminate known and unknown samples across decision thresholds.

Higher values are better.

AUPR
----

Area Under the Precision-Recall Curve.

AUPR provides a complementary evaluation of unknown-class detection, particularly when class frequencies are imbalanced.

Higher values are better.

FPR95
-----

False Positive Rate at 95% True Positive Rate.

FPR95 measures the false-positive rate when the detector reaches a true-positive rate of 95% for the positive/unknown class under the evaluation convention used in the notebook.

Lower values are better.


6. EXPLAINABILITY ANALYSIS
==========================

Objective
---------

The explainability analysis investigates which original input features contribute to the prototype-distance-based unknown score obtained from the Margin-Aware DANN representation.

SHAP
----

The project uses SHAP (SHapley Additive exPlanations) for feature-attribution analysis.

Depending on the experiment and scoring implementation, the notebook uses:

- shap.GradientExplainer
- shap.PermutationExplainer

The explanation target is constructed around the prototype-distance-based unknownness score.

Background/reference observations and explained samples are selected according to the experimental protocol implemented in the relevant notebook section.

Outputs
-------

The explainability workflow can generate:

- Global mean absolute SHAP importance
- Top-feature rankings
- Batch-specific feature importance
- Temporal feature-importance heatmaps
- Comparisons of feature relevance across domain shifts

These analyses are used to examine how the features contributing to prototype-based unknown detection vary across experimental conditions and temporal batches.


7. SOFTWARE ENVIRONMENT
=======================

Language:

- Python 3.x

Recommended environment:

- Google Colab

Main Python dependencies include:

- pandas
- numpy
- scipy
- scikit-learn
- tensorflow
- keras
- matplotlib
- seaborn
- shap
- joblib
- openpyxl

The Paderborn data preparation workflow additionally uses the unrar system utility for extracting the downloaded .rar archives.

Example installation in Google Colab:

!apt-get install -y unrar


8. GOOGLE DRIVE SETUP
=====================

The notebooks are designed to save intermediate and final outputs to Google Drive.

Google Drive can be mounted in Colab using:

from google.colab import drive
drive.mount('/content/drive')

Typical project directories include:

/content/drive/MyDrive/Paderborn_OSDA/
/content/drive/MyDrive/CBRN_Gas_OpenSet/

Depending on the notebook and experiment, these directories may contain:

- Processed datasets
- Experimental condition packages
- Model checkpoints
- Training histories
- Evaluation results
- Statistical analyses
- Figures
- SHAP outputs
- CSV files
- Excel files

Paths can be modified if a different Google Drive directory structure is used.


9. DATASET ACQUISITION
======================

9.1 Paderborn University Bearing Dataset
-----------------------------------------

The notebook includes code for downloading the selected Paderborn Bearing DataCenter archives and extracting the corresponding .mat files.

The original dataset remains subject to the terms specified by Paderborn University.


9.2 UCI Gas Sensor Array Drift Dataset
---------------------------------------

The Gas Sensor Array Drift Dataset can be obtained from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

The downloaded archive contains the batch .dat files used by the preprocessing pipeline.


10. RUNNING THE NOTEBOOKS
=========================

The notebooks are intended to be executed sequentially from top to bottom.

The general workflow is:

1. Mount Google Drive.
2. Install required dependencies and system utilities.
3. Download or load the datasets.
4. Parse and validate the raw data.
5. Perform dataset-specific preprocessing.
6. Construct source, adaptation/validation, and test subsets.
7. Standardize features according to the experimental protocol.
8. Define the open-set experimental conditions.
9. Train Standard DANN models.
10. Train Class-Aware DANN models.
11. Train Margin-Aware DANN models.
12. Construct known-class prototypes.
13. Calculate MSP, energy, and prototype-distance scores.
14. Calculate AUROC, AUPR, and FPR95.
15. Aggregate results across experimental conditions.
16. Perform statistical comparisons where implemented.
17. Run SHAP-based explainability analysis.
18. Generate tables and figures.

Model training and SHAP analysis can be computationally intensive. Google Colab is recommended for convenient execution and access to accelerated hardware when applicable.


11. GENERATED OUTPUTS
=====================

Depending on the notebook section, generated outputs include files in formats such as:

- .csv
- .xlsx
- .pkl
- model checkpoint files
- .png
- .pdf

Outputs include:

- Dataset validation summaries
- Experimental split definitions
- Model training histories
- Open-set performance tables
- Ablation comparisons
- Statistical test results
- PCA visualizations
- Open-set detection results
- SHAP feature-importance rankings
- Temporal feature-importance heatmaps


12. REPRODUCIBILITY
===================

The experimental pipeline includes measures intended to support reproducibility and reduce data leakage, including:

- Fixed random seeds where applicable
- Source-derived feature scaling
- Separation of known and unknown classes according to the open-set protocol
- Exclusion of the held-out unknown class from known-class training and prototype construction
- Separation of test data from model training
- Separate calibration/adaptation and test subsets
- Saved experimental-condition definitions
- Saved model checkpoints and evaluation outputs

Users reproducing the experiments should preserve the source/target and known/unknown separation implemented in the notebooks.


13. DATASET AVAILABILITY
========================

The datasets used in this project are publicly available.

Gas Sensor Array Drift Dataset:

UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

Dataset DOI:
https://doi.org/10.24432/C5RP6W

Paderborn University Bearing Dataset:

Bearing DataCenter, Paderborn University
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter

The DOI below refers to the associated benchmark publication, not to the dataset itself:

https://doi.org/10.36001/phme.2016.v3i1.1577


14. DATASET CITATIONS AND USAGE TERMS
=====================================

13.1 Paderborn University Bearing Dataset
------------------------------------------

The bearing data used in this project are obtained from the Bearing DataCenter of the Chair of Design and Drive Technology (KAt), Paderborn University, Germany.

Official dataset page:
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter

According to the official Bearing DataCenter page, the data are licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. The official page states that non-commercial academic use is allowed and that citation of the origin is required.

The Bearing DataCenter requests citation of the following publication:

Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016).
Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification.
European Conference of the Prognostics and Health Management Society, Bilbao, Spain.

Associated benchmark publication DOI:
https://doi.org/10.36001/phme.2016.v3i1.1577

The official Bearing DataCenter also requests attribution to the KAt-DataCenter / Chair of Design and Drive Technology, Paderborn University, as described on its website.

Users should consult the official Bearing DataCenter page for the applicable dataset terms and attribution requirements.


13.2 UCI Gas Sensor Array Drift Dataset
----------------------------------------

The gas-sensor data used in this project are obtained from the UCI Machine Learning Repository.

Official dataset page:
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

DOI:
https://doi.org/10.24432/C5RP6W

The citation currently provided by the UCI Machine Learning Repository is:

Vergara, A. (2012).
Gas Sensor Array Drift Dataset [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5RP6W

According to the UCI Machine Learning Repository, the dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.

The dataset is associated with the following publication:

Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012).
Chemical gas sensor drift compensation using classifier ensembles.
Sensors and Actuators B: Chemical, 166-167, 320-329.


15. METHODOLOGICAL REFERENCES
=============================

Domain-Adversarial Neural Networks
-----------------------------------

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016).
Domain-Adversarial Training of Neural Networks.
Journal of Machine Learning Research, 17(59), 1-35.


Maximum Softmax Probability / Baseline OOD Detection
-----------------------------------------------------

Hendrycks, D., & Gimpel, K. (2017).
A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.
International Conference on Learning Representations (ICLR).


Energy-Based Out-of-Distribution Detection
-------------------------------------------

Liu, W., Wang, X., Owens, J. D., & Li, Y. (2020).
Energy-based out-of-distribution detection.
Advances in Neural Information Processing Systems, 33, 21464-21475.


SHAP
----

Lundberg, S. M., & Lee, S.-I. (2017).
A Unified Approach to Interpreting Model Predictions.
Advances in Neural Information Processing Systems, 30.


16. LICENSING
=============

No separate software license is specified for the source code and Google Colab notebooks in this project.

The datasets used by the project retain the licenses and usage terms specified by their respective providers:

- Paderborn University Bearing Dataset:
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

- UCI Gas Sensor Array Drift Dataset:
  Creative Commons Attribution 4.0 International (CC BY 4.0).

The source code and notebooks do not modify the ownership, licensing, or usage terms of the original datasets.


17. CONTRIBUTIONS AND ISSUES
============================

Suggestions, corrections, and reports of reproducibility issues may be submitted through the repository's issue-tracking or contribution mechanism, if available.

When reporting an issue, useful information includes:

- A description of the problem
- The relevant notebook section or cell
- The complete error message
- Python and library versions
- Steps required to reproduce the issue

Potential extensions include additional domain-shift experiments, alternative open-set detection methods, additional baseline models, visualization improvements, and reproducibility enhancements.


18. NOTES
=========

This project is intended for research and reproducibility purposes.

Numerical results may vary slightly depending on factors such as random initialization, software-library versions, hardware, numerical precision, and sampling used in explainability analyses.

When using the datasets, users should follow the citation and usage requirements stated by the original data providers.
