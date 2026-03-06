# Machine learning and deep learning pipelines for decoding motor imagery (MI) from EEG signals using both classical spatial filtering approaches (CSP / FBCSP) and deep learning (EEGNet)

This repository contains the code used for the research presented in:

Machine and Deep Learning for Decoding EEG Motor Rhythms in BCI-based Neuro-Rehabilitation
Master Thesis – University of Ljubljana, Faculty of Electrical Engineering 

The work focuses on interpretable signal-processing pipelines for Brain-Computer Interfaces (BCI) and compares them with end-to-end deep learning models across both benchmark datasets and clinical EEG recordings from stroke patients. In addition to the publicly available datasets used for offline benchmarking, custom laboratory experiments were conducted at the Institute of Systems and Robotics (ISR), University of Coimbra, using the developed pipelines. These experiments aimed to evaluate the developed pipelines within a real-time Brain–Computer Interface (BCI) framework with neural feedback, targeting applications in neurorehabilitation.

## Overview
Brain-Computer Interfaces (BCIs) enable direct communication between the brain and external devices by decoding neural signals. In motor imagery BCI, users imagine movements (e.g., left hand or right hand), which modulates sensorimotor rhythms in EEG signals. These modulations can be detected and translated into control commands.

<img width="1236" height="624" alt="graph_bci_MI" src="https://github.com/user-attachments/assets/54ea3599-7bf6-473c-bb0d-45ae537c0b1d" />

This repository implements and compares two major paradigms:
1. Classical signal-processing pipelines
  * Common Spatial Patterns (CSP)
  * Filter Bank Common Spatial Patterns (FBCSP)
2. Deep learning models
  * EEGNet (compact CNN architecture for EEG)

The goal is to evaluate these approaches in both:
  * Standard BCI benchmark datasets
  * Clinical neuro-rehabilitation datasets

Once the classification pipeline was successfully developed and evaluated, it was integrated into custom experimental settings at the Institute of Systems and Robotics (ISR), with the goal of providing online neural feedback in the form of robotic hand movement or visual feedback (bar increase/decrease).


Motor imagery produces changes in sensorimotor rhythms (µ and β bands) through:

* Event-Related Desynchronization (ERD) during movement imagination
* Event-Related Synchronization (ERS) after the movement

These spectral patterns enable the decoding of user intent.

## Classical Pipeline: CSP / FBCSP

The classical approach relies on feature engineering combined with a lightweight classifier.
The pipeline includes:
1. Data loading and event labeling
2. Epoch extraction (0.5 – 2.0 s after cue onset)
3. Bandpass filtering
4. Spatial filtering using CSP or FBCSP
5. Feature selection
6. SVM classification
7. Cross-validation
<img width="1505" height="444" alt="image" src="https://github.com/user-attachments/assets/d5fd5112-22e5-4e02-a0e0-f16e6c321974" />

### Common Spatial Patterns (CSP)

CSP is a spatial filtering algorithm that maximizes variance differences between two classes.
It identifies spatial filters that highlight class-specific oscillatory activity.
For each EEG trial:
* covariance matrices are computed
* whitening transformation is applied
* eigenvalue decomposition extracts spatial filters
* log-variance features are used for classification


making the algorithm interpretable, computationally efficient and suitable for real-time BCIs.

### Filter Bank Common Spatial Patterns (FBCSP)

FBCSP extends CSP by analyzing multiple frequency bands.
Instead of a single band (8–30 Hz), the signal is decomposed into several sub-bands  (4 Hz intervals), CSP is applied independently to each band, and the resulting features are concatenated.

<img width="1721" height="789" alt="image" src="https://github.com/user-attachments/assets/b4599ede-04a1-40c2-bff6-59d931d9ea6f" />


### Feature Selection

Because FBCSP generates many features, a mutual information (MI) ranking algorithm is used to select the most discriminative ones.

The pipeline:
1. Compute MI between each feature and class label
2. Rank features by MI score
3. Select top-K features

<img width="1400" height="500" alt="MI_score_selection" src="https://github.com/user-attachments/assets/3131fe28-2753-47d1-bc7e-252be6410419" />
<img width="1536" height="762" alt="topoplots_selected_fbcsp" src="https://github.com/user-attachments/assets/023e8e61-264d-4bac-9309-92d309eeb2ba" />

To avoid artifacts dominating feature ranking, a noise exclusion buffer removes features with unrealistically high MI scores caused by muscular or ocular artifacts.

### Classification

Classification is performed using:

SVM (Support Vector Machine)
* Kernel: RBF
* C = 1
* γ = 0.01

Evaluation protocol:

5 × 5 cross-validation

Accuracy

Cohen’s κ

Confusion matrix

Two tasks are evaluated:

Movement vs Rest (relevant for further lab paradigm)
Left vs Right imagery (benchmark task)

## Deep Learning Pipeline: EEGNet

EEGNet replaces handcrafted feature extraction with end-to-end learning.

<img width="1506" height="439" alt="image" src="https://github.com/user-attachments/assets/9e933857-1419-45e2-90e4-06973233330c" />

Architecture components:

*Block 1:* Temporal Convolution

Learns frequency filters similar to band-pass filtering.

*Block 2:* Depthwise Spatial Convolution

Captures spatial relationships between electrodes.

*Block 3:* Separable Convolution

Extracts joint spatial-temporal features.

*Output Layer*

Dense layer + Softmax classifier.

Training configuration:

Optimizer: Adam
Learning rate: 0.001
Epochs: 200
Dropout: 0.25
Batch normalization

EEGNet is designed to work well with small EEG datasets, making it suitable for BCI research.
