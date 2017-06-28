# splitnet-wrn

TensorFlow implementation of SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization, ICML 2017
 
![SplitNet Concept](https://user-images.githubusercontent.com/13655756/27619160-8537abb6-5bfb-11e7-8854-2b5ee8be2312.png)

 - Juyong Kim\*(SNU), Yookoon Park\*(SNU), Gunhee Kim(SNU), and Sung Ju Hwang(UNIST) (\*: Equal contributions)

We propose a novel deep neural network that is both lightweight and effectively structured for model parallelization. Our network, which we name as *SplitNet*, automatically learns to split the network weights into either a set or a hierarchy of multiple groups that use disjoint sets of features, by learning both the class-to-group and feature-to-group assignment matrices along with the network weights. This produces a tree-structured network that involves no connection between branched subtrees of semantically disparate class groups. SplitNet thus greatly reduces the number of parameters and required computations, and is also embarrassingly model-parallelizable at test time, since the evaluation for each subnetwork is completely independent except for the shared lower layer weights that can be duplicated over multiple processors, or assigned to a separate processor. We validate our method with two different deep network models (ResNet and AlexNet) on two datasets (CIFAR-100 and ILSVRC 2012) for image classification, on which our method obtains networks with significantly reduced number of parameters while achieving comparable or superior accuracies over original full deep networks, and accelerated test speed with multiple GPUs.

## Prerequisite

1. TensorFlow
2. Train/val/test split of CIFAR-100 dataset(please run `python download_cifar100.py`)

## How To Run

```shell
# Clone the repo.
git clone https://github.com/dalgu90/splitnet-wrn.git
cd splitnet-wrn

# Download CIFAR-100 dataset and split train set into train/val
python download_cifar100.py

# Find grouping of deep(2-2-2) split of WRN-16-8
./group.sh

# Split and finetune
./split.sh

# To evaluate
./eval.sh
```

## Acknowledgement

This work was supported by Samsung Research Funding Center of Samsung Electronics under Project Number SRFC-IT150203.


## Authors

[Juyong Kim](http://juyongkim.com/)<sup>*1</sup>, Yookoon Park<sup>*1</sup>, [Gunhee Kim](http://www.cs.cmu.edu/~gunhee/)<sup>1</sup>, and [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>2</sup>

1: [Vision and Learning Lab](http://vision.snu.ac.kr/) @ Computer Science and Engineering, Seoul National University, Seoul, Korea  
2: [MLVR Lab](http://ml.unist.ac.kr/) @ School of Electrical and Computer Engineering, UNIST, Ulsan, South Korea  
\*: Equal contribution


## License
```
    MIT license
```    
If you find any problem, please feel free to contact to the authors. :^)
