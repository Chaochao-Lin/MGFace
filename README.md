# MGFace: Occlusion-aware Mask Guided Network for Face Recognition

MGFace is a mask guided network to deal with occlusion. The architecture is as follows.
![fig1](https://github.com/Chaochao-Lin/MGFace/blob/main/imgs/fig1.jpg)

### Training Command
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 train_mgface.py configs/swin_mgface
```

The complete code is currently under review and will be public soon.
