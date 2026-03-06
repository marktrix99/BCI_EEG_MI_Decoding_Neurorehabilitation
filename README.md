# Machine learning and deep learning pipelines for decoding motor imagery (MI) from EEG signals using both classical spatial filtering approaches (CSP / FBCSP) and deep learning (EEGNet)

This repository contains the code used for the research presented in:

Machine and Deep Learning for Decoding EEG Motor Rhythms in BCI-based Neuro-Rehabilitation
Master Thesis – University of Ljubljana, Faculty of Electrical Engineering 

The work focuses on interpretable signal-processing pipelines for Brain-Computer Interfaces (BCI) and compares them with end-to-end deep learning models across both benchmark datasets and clinical EEG recordings from stroke patients. In addition to the publicly available datasets used for offline benchmarking, custom laboratory experiments were conducted at the Institute of Systems and Robotics (ISR), University of Coimbra, using the developed pipelines. These experiments aimed to evaluate the developed pipelines within a real-time Brain–Computer Interface (BCI) framework with neural feedback, targeting applications in neurorehabilitation.
## Overview
Brain-Computer Interfaces (BCIs) enable direct communication between the brain and external devices by decoding neural signals. In motor imagery BCI, users imagine movements (e.g., left hand or right hand), which modulates sensorimotor rhythms in EEG signals. These modulations can be detected and translated into control commands.
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
