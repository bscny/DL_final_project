# SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy

**IMPORTANT**: This is the unofficial implementation of the CVPR 2023 accepted paper [SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy](https://cvpr.thecvf.com/virtual/2023/poster/21335), we selected this paper as our 2025 Deep Learning final project at Department, Computer science of NCCU

[[Final Presentation Slide](https://github.com/bscny/DL_final_project/blob/main/docs/reports/presentation_slide.pdf)] [[Final Report (TBD)]()] [[Project Proposal](https://github.com/bscny/DL_final_project/blob/main/docs/reports/proposal.pdf)] [[Our codes](https://github.com/bscny/DL_final_project/blob/main/src)]

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

![hyper_params](https://github.com/bscny/DL_final_project/blob/main/docs/pictures/hyper_params_setting.png)

Now, we take a look at the result after training, starting from *CIFAR-100* dataset

![CIFAR](https://github.com/bscny/DL_final_project/blob/main/docs/pictures/CIFAR_result.png)

Let's take a look at the *Food-101* dataset

![Food](https://github.com/bscny/DL_final_project/blob/main/docs/pictures/Food_result.png)

Below are the overall summary tables

![overall](https://github.com/bscny/DL_final_project/blob/main/docs/pictures/overall.png)

## Trained Models and Stats

These are the links to our google drive, feel free to download~
- For models trained on ***CIFAR-100***:
    - [DenseNet](https://drive.google.com/drive/folders/1z-Hqp7Rxt8nmt1KXemq3JsSrhQYqT8AG?usp=drive_link)
        - [DenseNet-121 baseline](https://drive.google.com/file/d/1-fC4-vZaQLxYU2-tbt6v922q77Q_zoMa/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1P_Sch9TwjgU9Sw8QpLDdQlLzcMRICVwN/view?usp=sharing)
        - [DenseNet-121 with SCConv](https://drive.google.com/file/d/1SghiS8y0kMBBAontbtqJ5R_FMUhAXvv3/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1y4X0e0poRfaHz1tVEgOrvUG5Wh0qBTs3/view?usp=drive_link)
    - [ResNet](https://drive.google.com/drive/folders/1y5swUecP6EH1Ilb90-bhlGEY_LFTYCvh?usp=drive_link)
        - [ResNet-50 baseline](https://drive.google.com/file/d/1_ScdySoTLJT1E_7bj5ZWA0MIvGGqfPJs/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/12t8fjEHJBCFku_H4ag9h6a1TlAoY1ZoE/view?usp=drive_link)
        - [ResNet-50 with SCConv](https://drive.google.com/file/d/1z5iU6tyHho8T42_lYo344mvwIKsDsESW/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1r9mjbXGMMh4XeGuCgr-LcENA_4ncktvQ/view?usp=drive_link)
- For models trained on ***Food-101***:
    - [DenseNet](https://drive.google.com/drive/folders/1P2uE7jn2yyLBPaVY_qqN7aGgRigP0l52?usp=drive_link)
        - [DenseNet-121 baseline](https://drive.google.com/file/d/1U5msrTc7nj0_tUDZkj3QK0W01tGzd0s-/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1qcWsbo9IHx6YzYI_fG_hemP3uiYbefam/view?usp=drive_link)
        - [DenseNet-121 with SCConv](https://drive.google.com/file/d/1ZewaVDlq5x6csAh_0DowNQxdgzLdHg_G/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1rHXGa25SidyP2N9Rs4Z6Eg877KXqZt_F/view?usp=drive_link)
    - [ResNet](https://drive.google.com/drive/folders/1KvNYn11eo1EZV8qfMRTAAOCcNokJ_gTz?usp=drive_link)
        - [ResNet-50 baseline](https://drive.google.com/file/d/1e9yNraXK9f_fFQy30HNzxDhLBUU2WvS5/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1tl4oSTQIQe0Jh6_7TfA8K0F6DdCTgsFf/view?usp=drive_link)
        - [ResNet-50 with SCConv](https://drive.google.com/file/d/1YIDMc80SuyeOK3xWqBw81weslIGPZigD/view?usp=drive_link), [the trained stats](https://drive.google.com/file/d/1mOaOPt8G2VNJuqunhCjFb3bfUO1DWHM-/view?usp=drive_link)

## Team Members

- 游宗諺
- 王冠智
- 林子齊
- 嚴聲遠