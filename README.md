# CTA_Transfomer

# 1. ViT small model replication
Replication of ViT small for the chest_pneumonia_xray
- The ViT small model: "hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", #Params (M): 22.1
- X-ray data: '/home/jay/Documents/chest_pneumonia_xray/chest_xray' (data loader should be ready to use)
- Previous work: [src/train_hrp.py](https://github.com/JZCambridge/vit-ct/blob/main/src/train_hrp.py)

# 2. Notes
1. Need to benchmark the ViT small without Lora for the chest_xray
