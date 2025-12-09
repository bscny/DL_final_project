# SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy

**IMPORTANT**: This is the unofficial implementation of the CVPR 2023 accepted paper [SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy](https://cvpr.thecvf.com/virtual/2023/poster/21335), we selected this paper as our 2025 Deep Learning final project at Department of Computer science of NCCU

[[Final Presentation Slide]](https://cvpr.thecvf.com/virtual/2023/poster/21335) [[Final Report (TBD)](https://cvpr.thecvf.com/virtual/2023/poster/21335)] [[Project Proposal](https://cvpr.thecvf.com/virtual/2023/poster/21335)]

**Note**: For the trained models in our experiment, Please see the below google drive link, we also provided the trained stats (e.g. loss curve)

## Table of content

- [Introduction](#introduction)
- [Take a Glance at the Result](#take-a-glance-at-the-result)
- [Trained Models and Stats](#trained-models-and-stats)
- [Team Members](#team-members)

## Introduction

In this section, we introduce you what we have done on this project briefly:
1. We understood the core concept behind the paper and started implementing it using `PyTorch`
2. We targeted the famous `ResNet50` and `DenseNet121` as the baseline model, at the same time, replace these model's bottleneck 3-by-3 blocks with SCConv block in the paper
3. After preparing both the baseline and modified models, we trained them on *CIFAR-100* and *Food-101* to see if SCConv is as useful in both Low and High resolution as the paper suggested
4. We made the final slides and report (using IEEE conference-template) to demonstrate our results, findings, and conclusions. The following section provides a quick view of it

## Take a Glance at the Result

The following are the hyper parameters we used in our experiment (**Note**: in order to run on our RTX 4090, we decreased the batch size for training DensNet related models to 64 instead of 128, which is the paper's setting)

![hyper_params]()

Now, we take a look at the result after training, starting from CIFAR-100 dataset

![CIFAR]()

Let's take a look at the Food-101 dataset

![Food]()

Below are the overall summary tables

![overall]()

## Trained Models and Stats

These are the links to our google drive, feel free to download~
- For models trained on ***CIFAR-100***:
    - [ResNet-50 baseline]()
    - [DenseNet-121 baseline]()
    - [ResNet-50 with SCConv]()
    - [DenseNet-121 with SCConv]()
- For models trained on ***Food-101***:
    - [ResNet-50 baseline]()
    - [DenseNet-121 baseline]()
    - [ResNet-50 with SCConv]()
    - [DenseNet-121 with SCConv]()

## Team Members

- 游宗諺
- 王冠智
- 林子齊
- 嚴聲遠