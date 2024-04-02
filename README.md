# Probabilistic Contrastive Learning for Long-Tailed Visual Recognition

This repository contains the Pytorch implementation of the T-PAMI 2024 paper "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition".

> **Probabilistic Contrastive Learning for Long-Tailed Visual Recognition**<br>
> [Chaoqun Du](https://scholar.google.com/citations?user=0PSKJuYAAAAJ&hl=en),
> [Yulin Wang](https://www.wyl.cool/),
> [Shiji Song](https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=en&oi=ao),
> [Gao Huang](https://www.gaohuang.net),

[![TPAMI](https://img.shields.io/badge/TPAMI2024-ProCo-green)](https://ieeexplore.ieee.org/abstract/document/10444057)
[![arXiv](https://img.shields.io/badge/arxiv-ProCo-blue)](https://arxiv.org/abs/2403.06726)


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

|  Method  |  Dataset      |Imbalance  Factor  |        Epochs  |           Top-1  Acc.(%)     |          Model    |
|  :----:  |  :----:       |:------:   |       :----:   |       :--------:  |      :---:       |
|  ProCo   |  CIFAR100-LT  |100        |       200      |       52.8        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/e9e47e54b40542529138/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1F5B4cuE1aMrShLxapslxlQ6iRj1lcDcK/view?usp=drive_link)
|  ProCo   |  CIFAR100-LT  |100        |       400      |       54.2        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/eed82aa8bd15430eb91a/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1fJlSaTl2Z74OgJXPWyOdEbqS1WwZDlh9/view?usp=drive_link)
|  ProCo   |  CIFAR100-LT  |50         |       200      |       57.1        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/106dc689c68d4bf29f22/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1yh2HZNcxxWaz7k5lSaNMuVxkUuLhHNfY/view?usp=drive_link)
|  ProCo   |  CIFAR100-LT  |10         |       200      |       65.5        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/2913d850a9344b9f9fa4/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1WTlq6YOKJ1HG9Asl9RtyQ9jwb5cpD3Pc/view?usp=drive_link)
|  ProCo   |  CIFAR10-LT   |100        |       200      |       85.9        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/6c88268eadb5413e8b98/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1s0luV1HkvSaJd0xkZ_FRyYYjRwTrasml/view?usp=drive_link)
|  ProCo   |  CIFAR10-LT   |50         |       200      |       88.2        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/55d8dedcece84431aab6/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1RdSTqChtWvc_iAubW0OMyQURCHcctYZn/view?usp=drive_link)
|  ProCo   |  CIFAR10-LT   |10         |       200      |       91.9        |      [Tsinghua   Cloud](https://cloud.tsinghua.edu.cn/f/2fe00aacba6b48a2b689/?dl=1)/[Google   Drive](https://drive.google.com/file/d/1bl7Ipq5kFgou6WszYHAAuO_qyRTRIgJE/view?usp=drive_link)


We also provide the tensorboard logs for the CIFAR experiments in the logs folder.



|  Method  |  Backbone     |            Dataset      |     Epochs  |     Top-1       Acc.(%)    |                                                                           Model                                                                                          |
|  :----:  |  :------:     |            :-----:      |     :----:  |     :--------:  |          :---:                                                                       |
|  ProCo   |  ResNet-50    |            ImageNet-LT  |     90      |     57.3        |          [Tsinghua                                                                   Cloud](https://cloud.tsinghua.edu.cn/f/65b8347a5c924802b3ea/?dl=1)/[Google                     Drive](https://drive.google.com/file/d/1hjG526DzgZcjV02eivx9bICzkhRduu4E/view?usp=drive_link)
|  ProCo   |  ResNeXt-50|  ImageNet-LT  |            90    |       58.0  |           [Tsinghua  Cloud](https://cloud.tsinghua.edu.cn/f/b79733cac1f345118fca/?dl=1)/[Google  Drive](https://drive.google.com/file/d/16Ux5sGZ0Rium7II7AdS2V3p8nAgkfP6m/view?usp=drive_link)
|  ProCo   |  ResNet-50    |            iNaturalist  2018  |       90    |           73.5       |                                                                           [Tsinghua                                                                                      Cloud](https://cloud.tsinghua.edu.cn/f/e152e5f89b8f43198c96/?dl=1)/[Google                     Drive](https://drive.google.com/file/d/1-5CjaNmoGUNoOa6FMv2DsiZ3iLIdzzJ3/view?usp=drive_link)
|  ProCo   |  ResNet-50    |            ImageNet-LT  |     180     |     57.8        |          [Tsinghua                                                                   Cloud](https://cloud.tsinghua.edu.cn/f/b2a4c15858da4bceb534/?dl=1)/[Google                     Drive](https://drive.google.com/file/d/1af9i5jzJpTXMLJbFsIxS1Obf0Hb0b-bN/view?usp=drive_link)

## Get Started

### Requirements

- python 3.9
- pytorch 1.12.1
- torchvision 0.13.1
- tensorboard 2.11
- scipy 1.9.3

Above environment is recommended, but not necessary. You can also use other versions of the packages.



### Training

By default, we use 1$*$RTX3090 GPU for CIFAR,  4$*$RTX3090 GPUs for ImageNet training and 8$*$A100 (40G) GPUs for iNaturalist2018 training. You can adjust the batch size according to your GPU memory.

```[bash]
bash sh/ProCo_CIFAR.sh ${dataset} ${imbalance_factor} ${epochs}
bash sh/ProCo_CIFAR.sh cifar100 0.01 200
bash sh/ProCo_CIFAR.sh cifar100 0.01 400
bash sh/ProCo_CIFAR.sh cifar100 0.02 200
bash sh/ProCo_CIFAR.sh cifar100 0.1  200
bash sh/ProCo_CIFAR.sh cifar10  0.01 200
bash sh/ProCo_CIFAR.sh cifar10  0.02 200
bash sh/ProCo_CIFAR.sh cifar10  0.1  200
```

```[bash]

bash sh/ProCo_ImageNetLT_R50_90epochs.sh
bash sh/ProCo_ImageNetLT_R50_180epochs.sh
bash sh/ProCo_ImageNetLT_X50_90epochs.sh
bash sh/ProCo_inat_R50_90epochs.sh
```

### Evaluation



For evaluation, you can run the following command:


```[bash]
bash sh/ProCo_CIFAR.sh cifar100 0.01 200 ${checkpoint_path}
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


