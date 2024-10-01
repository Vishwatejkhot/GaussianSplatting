import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Define the data path and load the CSV file
data_path = r'C:\Users\khotv\rp_aliyah_4d_004_dancing_BLD'
df = pd.read_csv(os.path.join(data_path, 'camera_parameters.csv'))

# Function to preprocess data
def preprocess_data(df, image_size):
    albedo_images = np.stack(df['albedo'].apply(lambda x: np.array(x).reshape(*image_size, 3)).values)
    depth_images = np.stack(df['depth'].apply(lambda x: np.array(x).reshape(*image_size, 3)).values)
    normal_images = np.stack(df['normal'].apply(lambda x: np.array(x).reshape(*image_size, 3)).values)
    positions = df[['Position_X', 'Position_Y', 'Position_Z']].values
    rotations = df[['Rotation_X', 'Rotation_Y', 'Rotation_Z']].values
    intrinsic_matrices = np.stack(df['K'].apply(lambda x: np.array(x).reshape((3, 3))).values)
    return albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices

# Function to normalize data
def normalize_data(albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices):
    albedo_images = albedo_images / 255.0
    depth_images = (depth_images - depth_images.min()) / (depth_images.max() - depth_images.min())
    normal_images = (normal_images - normal_images.min()) / (normal_images.max() - normal_images.min())
    return albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices

# Function to generate point clouds
def generate_point_clouds(depth_images, albedo_images, normal_images, positions, rotations, intrinsic_matrices):
    point_clouds = []
    for i in range(len(depth_images)):
        x_flat = np.tile(np.arange(64), 64)
        y_flat = np.repeat(np.arange(64), 64)
        depth_flat = depth_images[i].flatten()
        albedo_flat = albedo_images[i].flatten()
        normal_flat = normal_images[i].flatten()
        point_clouds.append((x_flat, y_flat, depth_flat, albedo_flat, normal_flat, intrinsic_matrices[i]))
    return point_clouds

# Function to apply Gaussian splatting
def apply_gaussian_splatting(point_clouds, sigma=1.0):
    splatted_point_clouds = []
    for point_cloud in point_clouds:
        x_flat, y_flat, depth_flat, albedo_flat, normal_flat, intrinsic_matrix = point_cloud

        depth = depth_flat.reshape((64, 64, 3))
        albedo = albedo_flat.reshape((64, 64, 3))
        normal = normal_flat.reshape((64, 64, 3))

        splat_depth = gaussian_filter(depth, sigma=sigma)
        splat_albedo = gaussian_filter(albedo, sigma=sigma)
        splat_normal = gaussian_filter(normal, sigma=sigma)

        depth_flat_splat = splat_depth.flatten()
        albedo_flat_splat = splat_albedo.flatten()
        normal_flat_splat = splat_normal.flatten()

        splatted_point_clouds.append((x_flat, y_flat, depth_flat_splat, albedo_flat_splat, normal_flat_splat, intrinsic_matrix))
    return splatted_point_clouds

# Function to prepare the training data
def prepare_training_data(splatted_point_clouds):
    X = []
    for point_cloud in splatted_point_clouds:
        x_flat, y_flat, depth_flat, albedo_flat, normal_flat, intrinsic_matrix = point_cloud
        combined_features = np.concatenate([depth_flat, albedo_flat, normal_flat], axis=0)
        intrinsic_flat = intrinsic_matrix.flatten()
        feature_vector = np.concatenate([combined_features, intrinsic_flat])
        X.append(feature_vector)
    return np.array(X)

#PSNR
def calculate_psnr(true_images, pred_images, max_pixel_value=1.0):
    mse = np.mean((true_images - pred_images) ** 2)
    if mse == 0:  
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Define the optimized PyTorch model with enhancements
class GaussianSplattingModel(nn.Module):
    def __init__(self, input_shape):
        super(GaussianSplattingModel, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, np.prod(input_shape))
        self.dropout = nn.Dropout(0.3)
        self.reshape = nn.Unflatten(1, input_shape)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)  # Add dropout
        x = torch.sigmoid(self.fc4(x))
        x = self.reshape(x)
        return x

if __name__ == "__main__":
    data = {
        'albedo': [np.random.rand(64, 64, 3) * 255 for _ in range(151)],
        'depth': [np.random.rand(64, 64, 3) for _ in range(151)],
        'normal': [np.random.rand(64, 64, 3) for _ in range(151)],
        'Position_X': np.random.rand(151),
        'Position_Y': np.random.rand(151),
        'Position_Z': np.random.rand(151),
        'Rotation_X': np.random.rand(151),
        'Rotation_Y': np.random.rand(151),
        'Rotation_Z': np.random.rand(151),
        'K': [np.random.rand(3, 3) for _ in range(151)]
    }
    df = pd.DataFrame(data)
    image_size = (64, 64)

    # Preprocess and normalize the data
    albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices = preprocess_data(df, image_size)
    albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices = normalize_data(albedo_images, depth_images, normal_images, positions, rotations, intrinsic_matrices)

    # Generate point clouds and apply Gaussian splatting
    point_clouds = generate_point_clouds(depth_images, albedo_images, normal_images, positions, rotations, intrinsic_matrices)
    splatted_point_clouds = apply_gaussian_splatting(point_clouds, sigma=1.0)
    training_data = prepare_training_data(splatted_point_clouds)
    
    # Split data into training and test sets
    train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_tensor, train_data_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_data_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    input_shape = training_data.shape[1:]
    model = GaussianSplattingModel(input_shape)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for batch_x, _ in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
    model.eval()
with torch.no_grad():
    test_loss = 0
    all_preds = []
    all_targets = []
    psnr_list = []

    for batch_x, _ in test_dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        test_loss += loss.item()

        
        preds = outputs.numpy()
        targets = batch_x.numpy()

        all_preds.append(preds)
        all_targets.append(targets)

        
        batch_psnr = calculate_psnr(targets, preds)
        psnr_list.append(batch_psnr)

    # Calculate the average PSNR over all batches
    avg_test_loss = test_loss / len(test_dataloader)
    avg_psnr = np.mean(psnr_list)

    # Overall RMSE calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = sqrt(mse)

    print(f'Average Test Loss: {avg_test_loss:.4f}, RMSE: {rmse:.4f}, Average PSNR: {avg_psnr:.2f} dB')

    # Save the model
    torch.save(model.state_dict(), 'gaussian_splatting_model_optimized.pth')
    print("Model training complete and saved as 'gaussian_splatting_model_optimized.pth'")



