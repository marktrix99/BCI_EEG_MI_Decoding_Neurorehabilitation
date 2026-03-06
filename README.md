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
