## Semi-supervised Graph Convolutional Networks for the Domain Adaptive Recognition of Thyroid Nodules in Cross-Device Ultrasound Images

## Overview

**Background**:Ultrasound plays a critical role in the early screening and diagnosis of cancers. Although deep neural networks have been widely investigated in the computer-aided diagnosis (CAD) of different medical images, diverse ultrasound devices and image modalities pose challenges for clinical applications, especially in the recognition of thyroid nodules having various shapes and sizes. More generalized and extensible methods need to be developed for the cross devices recognition of thyroid nodules.

**Purpose**:In this work, a semi-supervised graph convolutional deep learning framework is proposed for the domain adaptative recognition of thyroid nodules across several ultrasound devices. A deep classification network, trained on a source domain with a specific device, can be transferred to recognize thyroid nodules on target domain with other devices, using only few manual annotated ultrasound images.

**Methods**:This study presents a semi-supervised graph-convolutional-network-based domain adaptation framework, namely Semi-GCNs-DA. Based on the ResNet backbone, it is extended in three aspects for domain adaptation, i.e., graph convolutional networks (GCNs) for the connection construction between source and target domains, semi-supervised GCNs for accurate target domain recognition, and pseudo labels for unlabeled target domains. Data were collected from 1,498 patients comprising 12,108 images with or without thyroid nodules and using three different ultrasound devices. Accuracy, Sensitivity and Specificity were used for the performance evaluations.

**Results**:The proposed method was validated on 6 groups of data for a single source domain adaptation task, the mean Accuracy was 0.9719±0.0023, 0.9928±0.0022, 0.9353±0.0105, 0.8727±0.0021, 0.7596±0.0045, 0.8482±0.0092, which achieved better performance in comparison with the state-of-the-art. The proposed method was also validated on 3 groups of multiple source domain adaptation tasks. In particular, when using X60 and HS50 as the source domain data, and H60 as the target domain, it can achieve the Accuracy of 0.8829 ± 0.0079, Sensitivity of 0.9757 ± 0.0001, and Specificity of 0.7894 ± 0.0164. Ablation experiments also demonstrated the effectiveness of the proposed modules.

## Setup Environment

For this project, we used python 3.7.3, torch 1.7.1+cu110, torch-geometric 2.2.0.

All experiments were executed on a NVIDIA RTX 2080 Ti.


## Setup Datasets

The data set is divided by txt file. The txt file is stored in the folder named txt. In order to use a larger batch size, the data from file named targetReal is resampled. In other words, there are many duplicate samples in this file.


## Training

A training job can be launched using:

```shell
python main.py
```

