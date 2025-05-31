# CTA_Transformer

A Vision Transformer (ViT) implementation for chest X-ray pneumonia classification and coronary CT angiography (CTA) analysis.

## Overview

This project implements a Vision Transformer architecture with several key features:

- Vision Transformer (ViT) models of various sizes (tiny, small, base, large)
- Support for LoRA (Low-Rank Adaptation) fine-tuning
- Hybrid architecture combining ConvNeXt and Transformer components
- Cross-validation support
- Extensive data augmentation pipeline
- Multi-class and binary classification capabilities

## Installation
```bash
git clone https://github.com/JZCambridge/vit-ct.git
cd vit-ct
pip install -r requirements.txt
```

## Model Architecture

The repository implements several model variants:

1. **Pure ViT**: Standard Vision Transformer implementation
2. **Hybrid Res-ViT**: Combines ConvNeXt features with Transformer architecture
3. **ConvNeXt Blocks**: Enhanced convolution blocks with vessel-aware attention

## Usage

### Training

```bash
python main.py \
    -event_name "experiment_name" \
    -vit_size "small" \
    -bs 32 \
    -data_path "/path/to/data" \
    -epochs 60 \
    -train_type "resvit" \
    -loss_func "bce"
```


Key parameters:
- `event_name`: Name of the experiment
- `vit_size`: Size of ViT model (base, small, tiny, large)
- `bs`: Batch size
- `data_path`: Path to dataset
- `epochs`: Number of training epochs
- `train_type`: Training type (lora, full, resnet, net, resvit)
- `loss_func`: Loss function (bce, ce, ca3, crps, focal)

### Data Loading

The repository supports multiple data loading configurations:

- Random splitting
- Stratified sampling  
- Cross-validation
- Custom augmentations

### Evaluation Metrics

The model evaluation includes comprehensive metrics:

- Accuracy
- AUC-ROC
- Sensitivity/Specificity
- F1 Score
- Precision/Recall

## Data Augmentation

Supports various augmentation techniques:

- Random flipping (height/width/diagonal)
- Rotation
- Gaussian noise
- Frame swinging
- Column shifting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.