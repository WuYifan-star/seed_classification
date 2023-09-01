# seed_classification

## introduction

This is a PyTorch implementation of oil palm seeds quality classification. This repo is implemented in PyTorch and GPU.  

<div  align="center">    
<img src="https://github.com/WuYifan-star/seed_classification/assets/56722065/a52dab3c-fc5f-4616-8249-217f58a68a34" width = "600"  alt="structure" />
</div>

## Main Results

The following results are based on the oil palm seeds dataset(collected by Advanced Agricultural Resources (AAR) Sdn. Bhd.).

<div  align="center">    
<img src="https://github.com/WuYifan-star/seed_classification/assets/56722065/102041b5-a796-4f78-a193-7803b2b34a80" width = "600"  alt="structure"  />
</div>

## Usage: Preparation

Install PyTorch and download the oil palm seeds dataset.  

The code has been tested with CUDA 11.6, PyTorch 1.11.0 and timm 0.6.13.

The code needs support from the grad-cam repo(https://github.com/jacobgil/pytorch-grad-cam) and dino repo(https://github.com/facebookresearch/dino/tree/main).

## Usage: 

First step: Run the Train_net.ipynb to get the  trained_models, and you can find the training log in the log fold. You can change the model structure in the model.py.   

Second step: Run the evaluate.ipynb.

Third step: Run the saliency_map.ipynb.

## contact details:

wuyifan736@gmail.com
