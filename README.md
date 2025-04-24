# AlexNet vs VGG-16: Comparative Analysis on CIFAR-10

This repository contains a comprehensive comparative study of two foundational CNN architectures: AlexNet and VGG-16, implemented and evaluated on the CIFAR-10 dataset. The project demonstrates the architectural differences, implementation approaches, and performance trade-offs between these influential deep learning models.

## Project Overview

This project explores:
- Detailed architectural analysis of AlexNet and VGG-16
- Implementation of both architectures using PyTorch
- Adaptation strategies for running ImageNet-designed models on smaller CIFAR-10 images
- Training performance metrics and visualizations
- Receptive field analysis and growth patterns
- Implementation of VGG-16 with Batch Normalization as an enhancement

## Repository Contents

- `AlexNet.ipynb`: Implementation and training of the AlexNet architecture
- `VGG.ipynb`: Implementation and training of the VGG-16 architecture
- `Comparison.ipynb`: Direct comparison between both models including performance metrics and visualizations
- `Report.pdf`: Detailed analysis of architecture designs, theoretical underpinnings, and experimental results

## Key Findings

| Model | Time per Epoch (min) | Final Test Accuracy (%) |
|-------|---------------------|-------------------------|
| AlexNet | 3.73 | 84.15 |
| VGG-16 | 21.0 | 90.38 |
| VGG-16 with BN | ~21.3 | 91.00 |

### Performance Highlights:
- **VGG-16** outperforms **AlexNet** by ~6.23% in accuracy at the cost of ~5.6x longer training time
- **Batch Normalization** further improves VGG-16 performance by 0.62% (representing a 6.4% relative error reduction)

## Implementation Details

### Dataset Preparation
The CIFAR-10 dataset (32×32 color images) was processed to work with architectures designed for larger images:
- Resizing to 256×256
- Center cropping to 224×224
- Normalization using ImageNet mean and standard deviation

### Training Configuration
- Batch size: 4
- Optimizer: SGD with momentum (0.7)
- Learning rate: 0.001
- Loss function: Cross-Entropy Loss
- Epochs: 4
- Hardware: CUDA-compatible GPU

### Architecture Modifications
Both architectures were modified to accommodate the 10-class CIFAR-10 dataset (from ImageNet's 1000 classes):
- **AlexNet**: Final FC layer modified to output 10 classes
- **VGG-16**: Added intermediate FC layer (4096→512) and modified output layer (512→10)
- **VGG-16 with BN**: Added BatchNorm2d layers after each convolution

## Architectural Comparison

### AlexNet
- 8 layers deep (5 convolutional, 3 fully connected)
- ~61M parameters
- Varied filter sizes (11×11, 5×5, 3×3)
- Rapid receptive field growth
- ReLU activation and Dropout regularization

### VGG-16
- 16 layers deep (13 convolutional, 3 fully connected)
- ~138M parameters
- Consistent 3×3 filters throughout
- Gradual receptive field growth
- More controlled feature extraction

## Receptive Field Analysis
The report includes a detailed analysis of how the receptive field grows throughout each network:
- **AlexNet**: Rapid initial growth due to large filters and strides
- **VGG-16**: More gradual and controlled growth pattern through stacked small filters

## Conclusion
The experiments confirm that deeper architectures with appropriate regularization techniques can extract more discriminative features from image data. VGG-16's superior performance over AlexNet demonstrates the effectiveness of architectural depth and consistent filter design, while the further improvement achieved with Batch Normalization highlights the importance of normalization techniques in deep network training.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- CUDA-compatible GPU (recommended)


