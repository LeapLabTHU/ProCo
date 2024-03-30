# Probabilistic Contrastive Learning for Long-Tailed Visual Recognition

This repository contains the Pytorch implementation of the T-PAMI 2024 paper "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition".

> **Probabilistic Contrastive Learning for Long-Tailed Visual Recognition**<br>
> [Chaoqun Du](https://scholar.google.com/citations?user=0PSKJuYAAAAJ&hl=en),
> [Yulin Wang](https://www.wyl.cool/),
> [Shiji Song](https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=en&oi=ao),
> [Gao Huang](https://www.gaohuang.net),

[![TPAMI](https://img.shields.io/badge/TPAMI2024-ProCo-green)](https://ieeexplore.ieee.org/abstract/document/10444057)
[![arXiv](https://img.shields.io/badge/arxiv-ProCo-blue)](hhttps://arxiv.org/abs/2403.06726)


## Introduction

<p align="center">
    <img src="figures/1.png" width= "420">
</p>

We proposed a novel probabilistic contrastive (ProCo) learning algorithm for long-tailed distribution.
Specifically, we employed a reasonable and straight-forward von Mises-Fisher distribution to model the normalized feature space of samples in the context of contrastive learning. This choice offers two key advantages.
First, it is efficient to estimate the distribution parameters across different batches by maximum likelihood estimation.
Second, we derived a closed form of expected supervised contrastive loss for optimization by sampling infinity samples from the estimated distribution.
This eliminates the inherent limitation of supervised contrastive learning that requires a large number of samples to achieve satisfactory performance.


## Results

### Supervised Image Classification

| Method | Backbone | Dataset | Epochs | Top-1 Acc.(%) | Model |
| :----: | :------: | :-----: | :----: | :--------: | :---: |
| ProCo | ResNet-50 | ImageNet-LT | 90 | 57.3  | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/65b8347a5c924802b3ea/?dl=1)
| ProCo | ResNeXt-50| ImageNet-LT | 90 | 58.0 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b79733cac1f345118fca/?dl=1)
| ProCo | ResNet-50 | iNaturalist 2018 | 90 | 73.5 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e152e5f89b8f43198c96/?dl=1)
| ProCo | ResNet-50 | ImageNet-LT | 180 | 57.8 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b2a4c15858da4bceb534/?dl=1)


## Get Started

### Requirements

- python 3.9
- pytorch 1.12.1
- torchvision 0.13.1
- tensorboard 2.11
- scipy 1.9.3

Above environment is recommended, but not necessary. You can also use other versions of the packages.



### Training

```[bash]
bash sh/ProCo_ImageNetLT_R50_90epochs.sh
bash sh/ProCo_ImageNetLT_R50_180epochs.sh
bash sh/ProCo_ImageNetLT_X50_90epochs.sh
bash sh/ProCo_inat_R50_90epochs.sh
```

### Evaluation

For example, if you want to evaluate the model trained with 90 epochs on ImageNet-LT, you can run the following command:

```[bash]
bash sh/ProCo_ImageNetLT_R50_90epochs.sh ${checkpoint_path}
```

## ToDo

- [ ] Long-tailed Semi-Supervised Learning.

## Citation

If you find this code useful, please consider citing our paper:

```[tex]
@article{du2024probabilistic,
  title={Probabilistic Contrastive Learning for Long-Tailed Visual Recognition},
  author={Du, Chaoqun and Wang, Yulin and Song, Shiji and Huang, Gao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

## Contact

If you have any questions, please feel free to contact the authors. Chaoqun Du: <dcq20@mails.tsinghua.edu.cn>.

## Acknowledgement

Our code is based on the BCL (Balanced Contrastive Learning for Long-Tailed Visual Recognition) repository.


