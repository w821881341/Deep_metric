# Deep Metric Learning

### Learn a deep metric which can be used image retrieval , clustering.
============================

## Pytorch Code for deep metric methods:

- Contrasstive Loss

- Semi-Hard Sampling 

    Sampling strategy in FaceNet 

- Lifted Structure Loss (I modify this loss because of its weak performance of original lifted structure loss)

Deep Metric Learning via Lifted Structured Feature Embedding
[](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)

- Binomial BinDeviance Loss 

Deep Metric Learning for Person Re-Identification [](http://www.cbsr.ia.ac.cn/users/zlei/papers/ICPR2014/Yi-ICPR-14.pdf)

- Distance Weighted Sampling

Sampling Matters in Deep Embedding Learning [](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf)

- NCA Loss

   Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure  -Ruslan Salakhutdinov and Geoffrey Hinton
 
 -WeightLoss 
    My own loss (not public now)
 

## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set

- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online-Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing

- [In-Shop-clothes-Retrieval](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
    For the In-Shop Clothes Retrieval dataset, 3,997 classes with 25,882 images for training.
    And the test set are partitioned to query set with 3,985 classes(14,218 images) and gallery set with 3,985 classes (12,612 images).


## Prerequisites

- Pytorch 1.0
- Computer with Linux or OSX
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.
 
### performance on CUB-200 and Cars-196

|Recall@K | 1 | 2 | 4 | 8 | 16 | 32 | 1 | 2 | 4 | 8 | 16 | 32|
 |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|HDC | 53.6 | 65.7 | 77.0 | 85.6 | 91.5 | 95.5 | 73.7 | 83.2 | 89.5 | 93.8 | 96.7 | 98.4|
|Clustering | 48.2 | 61.4 | 71.8 | 81.9 | - | - | 58.1 | 70.6 | 80.3 | 87.8 | - | -|
|ProxyNCA | 49.2 | 61.9 | 67.9 | 72.4 | - | - | 73.2 | 82.4 | 86.4 | 87.8 | - | -|
|Smart Mining | 49.8 | 62.3 | 74.1 | 83.3 | - | - | 64.7 | 76.2 | 84.2 | 90.2 | - | -|
|Margin | 63.6| 74.4| 83.1| 90.0| 94.2 | - | 79.6| 86.5| 91.9| 95.1| 97.3 | - |
|HTL | 57.1| 68.8| 78.7| 86.5| 92.5| 95.5 | 81.4| 88.0| 92.7| 95.7| 97.4| 99.0 |
|ABIER |57.5 |68.7 |78.3 |86.2 |91.9 |95.5 |82.0 |89.0 |93.2 |96.1 |97.8 |98.7|
|Weight|  66.85|  77.84|  85.8|   91.29 |  94.94 |  97.42 |  83.69| 90.27 |  94.53|  97.16 |  98.65 |  99.36|

###  performance on SOP and In-shop 

|Recall@K | 1 | 10 | 100 | 1000 | 1 | 10 | 20 | 30 | 40 | 50|
 |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Clustering | 67.0 | 83.7 | 93.2 | - | -| -| -| -| - | -|
|HDC | 69.5 | 84.4 | 92.8 | 97.7 | 62.1 | 84.9 | 89.0 | 91.2 | 92.3 | 93.1|
|Margin | 72.7 | 86.2 | 93.8 | 98.0 | -| -| - | -| -| -|
|Proxy-NCA | 73.7 | - | - | - | -| -| - | - | -| -|
|ABIER | 74.2 | 86.9 | 94.0 | 97.8 | 83.1 | 95.1 | 96.9 | 97.5 | 97.8 | 98.0|
|HTL | 74.8| 88.3| 94.8| 98.4 | 80.9| 94.3| 95.8| 97.2| 97.4| 97.8 ||
|weight |  78.18|  90.47|  96.0|  98.74 |89.64 |97.87|98.47|98.84 |99.05 |99.20|


##### Reproducing Car-196 (or CUB-200-2011) experiments with dimension 512 

*** weight :***

```bash
sh run_train_00.sh
```
