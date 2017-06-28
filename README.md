# splitnet-wrn

TensorFlow implementation of SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization, ICML 2017

 - Juyong Kim\*(SNU), Yookoon Park\*(SNU), Gunhee Kim(SNU), and Sung Ju Hwang(UNIST) (*: Equal contributions)

<b>Prerequisite</b>

1. TensorFlow
2. Train/val/test split of CIFAR-100 dataset(please run `python download_cifar100.py`)

<b>How To Run</b>

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
