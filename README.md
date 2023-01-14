## Introduction to the code (Using transformer in pytorch)

Transformer(Attention is all you need, ViT): https://arxiv.org/abs/1706.03762v5, https://openreview.net/pdf?id=YicbFdNTTy

if you wanan use this code
In dataset, you can use up.py to interpolate the data, and use data_cut to split the data into training and test sets.

In the training phase, if there is only one GPU, use train_single_gpu.py

If there are more than one, use train_multi_gpu_using_launch.py 
# Enter the following code sample into the terminal
# CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_single_gpu.py
# CUDA_VISIBLE_DEVICES is number the graphics card        nproc_per_node is the number of graphics cards used
