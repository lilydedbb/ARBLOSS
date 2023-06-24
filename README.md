# ARBLOSS

**Neural Collapse Inspired Attraction-Repulsion-Balanced Loss for Imbalanced Learning**

**Authors**: Liang Xie, Yibo Yang, Deng Cai, Xiaofei He

[[`pre-print version`](https://arxiv.org/abs/2204.08735)]

[[`published version`]](https://www.sciencedirect.com/science/article/abs/pii/S0925231223000309)

<div align="center">
  <img src="./assets/MiSLAS.PNG" style="zoom:90%;"/>
</div><br/>

**Introduction**: This repository provides an implementation for our paper "[Neural Collapse Inspired Attraction-Repulsion-Balanced Loss for Imbalanced Learning](https://arxiv.org/abs/2204.08735)" published on [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing). This repository is based on [MiSLAS](https://github.com/dvlab-research/MiSLAS). ARBLOSS can achieve state-of-the-art performance via only one-stage training instead of 2-stage learning like nowadays SOTA works.

## Installation

You can refer the instructions in [MiSLAS](https://github.com/dvlab-research/MiSLAS) to install the environments.

> **Requirements**
> 
> * Python 3.7
> * torchvision 0.4.0
> * Pytorch 1.2.0
> * yacs 0.1.8
> 
> **Virtual Environment**
> ```
> conda create -n ARBLOSS python==3.7
> source activate ARBLOSS
> ```
> 
> **Install MiSLAS**
> ```
> git clone https://github.com/lilydedbb/ARBLOSS.git
> cd ARBLOSS
> pip install -r requirements.txt
> ```
> 
> **Dataset Preparation**
> * [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
> * [ImageNet](http://image-net.org/index)
> * [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
> * [Places](http://places2.csail.mit.edu/download.html)
> 
> Change the `data_path` in `config/*/*.yaml` accordingly.

## Training

ARBLOSS only using the stage-1 training of the code base.

Trainging on CIFAR10/100:

```
python train_stage1.py --cfg ./config/cifar100/cifar100_imb01_stage1_mixup.yaml
python train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml
python train_stage1.py --cfg ./config/cifar100/cifar100_imb002_stage1_mixup.yaml

python train_stage1.py --cfg ./config/cifar10/cifar10_imb01_stage1_mixup.yaml
python train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup.yaml
python train_stage1.py --cfg ./config/cifar10/cifar10_imb002_stage1_mixup.yaml
```

Training on ImageNet-LT:

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 24389 \
       train_stage1.py --cfg ./config/imagenet/imagenet_resnet50_arbloss_ugcc_mixup.yaml
```

Training on iNaturalist2018:

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 24389 \
       train_stage1.py --cfg ./config/ina2018/ina2018_resnet50_arbloss_ugcc_mixup.yaml
```

Training on Places-LT:

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 24389 \
       train_stage1.py --cfg ./config/places/places_resnet50_arbloss_ugcc_mixup.yaml
```

<!--## Results and Models-->

## <a name="Citation"></a>Citation

Please consider citing ARBLOSS in your publications if it helps your research. :)

```bib
@article{XIE202360,
title = {Neural collapse inspired attraction–repulsion-balanced loss for imbalanced learning},
journal = {Neurocomputing},
volume = {527},
pages = {60-70},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.01.023},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223000309},
author = {Liang Xie and Yibo Yang and Deng Cai and Xiaofei He},
keywords = {Long-tailed learning, Neural collapse, Machine Learning, Image Classification},
abstract = {Class imbalance distribution widely exists in real-world engineering. However, the mainstream optimization algorithms that seek to minimize error will trap the deep learning model in sub-optimums when facing extreme class imbalance. It seriously harms the classification precision, especially in the minor classes. The essential reason is that the gradients of the classifier weights are imbalanced among the components from different classes. In this paper, we propose Attraction–Repulsion-Balanced Loss (ARB-Loss) to balance the different components of the gradients. We perform experiments on large-scale classification and segmentation datasets, and our ARB-Loss can achieve state-of-the-art performance via only one-stage training instead of 2-stage learning like nowadays SOTA works.}
}
```

## Contact

If you have any questions about this work, feel free to contact us through email (Liang Xie: lilydedbb@gmail.com) or Github issues.
