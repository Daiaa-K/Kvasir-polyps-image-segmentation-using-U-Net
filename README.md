# Kvasir-SEG Polyp Image Segmentation

## Project Overview
This project implements a U-Net architecture for automatic segmentation of polyps in colonoscopy images using the Kvasir-SEG dataset. Early detection of polyps is crucial for preventing colorectal cancer, and this solution aims to improve polyp detection rates through deep learning.

## About the Dataset
The Kvasir-SEG dataset contains 1000 polyp images and their corresponding ground truth masks from the Kvasir Dataset v2. Image resolutions vary from 332×487 to 1920×1072 pixels.

### Medical Context
Polyps are precursors to colorectal cancer and are found in nearly half of individuals at age 50 undergoing screening colonoscopy. Current polyp miss rates during colonoscopies range from 14%-30% depending on polyp type and size. Automated detection systems can play a crucial role in improving both prevention and survival rates.

## Model Architecture
The implementation uses a U-Net model with:
- **Encoder**: 5 blocks of convolutional layers with max pooling
- **Decoder**: 5 blocks of transpose convolutions with skip connections
- **Regularization**: BatchNormalization layers
- **Initialization**: He normal for weights
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy and Binary IoU (Intersection over Union)

## Implementation Details
- Images are resized to 256×256 pixels
- Data preprocessing includes normalization and mask binarization
- Model training includes callbacks for:
  - Model checkpointing (saves best model based on validation accuracy)
  - Learning rate reduction (adjusts learning rate when validation loss plateaus)

## Repository Structure
```
├── ksavir-polyps-image-segmentation.ipynb
├── kvasir-seg/
│   └── Kvasir-SEG/
│       ├── images/
│       └── masks/
└── README.md
```

## How to Use
1. Clone the repository
2. Install dependencies:
   ```
   pip install tensorflow opencv-python matplotlib seaborn numpy pandas plotly scikit-learn
   ```
3. Download the Kvasir-SEG dataset and place it in the proper directory
4. Run the notebook cells sequentially
5. To predict on new images, use the provided prediction code sections

## Results
The notebook includes visualizations comparing:
- Original colonoscopy images
- Ground truth polyp masks
- Model predictions
- Overlay visualizations showing predictions on original images

The model performs well on both the validation set and unseen test images.

## Future Work
- Experiment with more advanced architectures (DeepLabv3+, SegNet)
- Implement additional data augmentation techniques
- Test on diverse datasets for better generalization
- Optimize for potential real-time usage in clinical settings
- Explore model interpretability for clinical validation

## Dependencies
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Plotly
- Scikit-learn


## Acknowledgments
- This work uses the Kvasir-SEG dataset from the Kvasir Dataset v2
