# Volumetric Capture and 3D Model Optimization using NeRF and Gaussian Splatting

This repository contains code and resources related to my internship project at **Abhinaya Ltd**, where I explored advanced techniques for volumetric capture and 3D model reconstruction. The project involved using **NeRF** (Neural Radiance Fields) and **Gaussian Splatting** to capture and render high-quality 3D scenes using a multicamera setup.

## Project Overview

### 1. Initial Setup
- **Volumetric Video Basics**: Learned the fundamentals of volumetric capture through recommended videos and tutorials on NeRF and Gaussian Splatting.
- **Environment**: Set up a development environment using tools like Blender, Python, and necessary libraries for NeRF and Gaussian Splatting.

### 2. Dataset Preparation
- **4D People Dataset**: Downloaded from Renderpeople and used in Blender for simulating a multicamera system.
- **Multicamera Capture**: Cameras were set up at angles like 0°, 45°, 90°, and more, capturing RGB images, depth maps, and normal maps at different frame rates.

### 3. Baseline Evaluation with NeRF
- **NeRF Pipeline**: The `nerfstudio` repository was used to train the NeRF model with different camera setups and configurations.
- **Metrics**: Evaluated using SSIM and PSNR to measure the quality of the 3D volume encoding.

### 4. Gaussian Splatting Integration
- **Modified NeRF Pipeline**: The pipeline was adapted to integrate Gaussian Splatting for improved performance.
- **Training**: Used the same dataset and configurations for training the Gaussian Splatting model.

### 5. Performance Evaluation
- **Comparison**: Compared NeRF and Gaussian Splatting models using the same metrics.
- **Time Offset Evaluation**: Tested varying capture timings across cameras to analyze synchronization impacts.

## Results

The final optimized Gaussian Splatting model achieved the following results:

- **Average Test Loss**: 0.0032
- **RMSE**: 0.0569
- **Average PSNR**: 24.89 dB

The trained model has been saved as `gaussian_splatting_model_optimized.pth`.

## Future Goals

Moving forward, the following goals are set to improve the model's performance:

- **Parameter Tuning**: Adjusting hyperparameters to increase PSNR and improve visual fidelity.
- **Optimization**: Exploring advanced optimization techniques for better accuracy and reduced artifacts.
- **Real-time Performance**: Working to optimize the model for faster inference in real-time applications.
- **Dataset Expansion**: Testing with more diverse datasets to increase robustness.

## Setup Instructions

### Requirements
- Python 3.8+
- Blender
- Required libraries: `nerfstudio`, `PyTorch`, `NumPy`, etc.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/volumetric-capture-3d.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the 4D People Dataset from Renderpeople and set up the camera system in Blender as described in the project.

4. To run the NeRF pipeline, use:
    ```bash
    python train_nerf.py --config nerf_config.json
    ```

5. To run the Gaussian Splatting model:
    ```bash
    python train_gaussian_splatting.py --config gaussian_config.json
    ```

## Usage

- Use the pre-trained models or train your own using the provided dataset configuration.
- Adjust the camera setup or hyperparameters in the config files to experiment with different setups.
- Evaluate the models using the provided scripts to measure SSIM and PSNR.

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to **Abhinaya Ltd** for the opportunity to work on this exciting project.
- Resources used: Renderpeople's 4D People Dataset, nerfstudio repository for NeRF, and various research papers on Gaussian Splatting.
