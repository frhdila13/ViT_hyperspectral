import os
import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data
import tifffile as tiff
from scipy import io

# --- Configuration Registry ---
# This acts as a central menu for all your datasets.
DATASETS_CONFIG = {
    'sa': {
        'img': 'Salinas_corrected.mat', 
        'gt': 'Salinas_gt.mat', 
        'img_key': 'salinas_corrected', 
        'gt_key': 'salinas_gt'
    },
    'pu': {
        'img': 'PaviaU.mat', 
        'gt': 'PaviaU_gt.mat', 
        'img_key': 'paviaU', 
        'gt_key': 'paviaU_gt'
    },
    'chikusei': {
        'img': 'subset_hyper_Chikusei.tif', 
        'gt': 'Rasterized_chikusei_GT.tif'
    },
    'enmap': {
        'img': 'enmap_petaling.tif'
        'gt': 'Rasterized_GT_enmap_petaling.tif'
    },
    'my_project': {
        'img': 'data.tif', 
        'gt': 'labels.tif'
    }
}

def load_hsi(dataset_name, dataset_dir):
    """ Universal HSI loader for both .mat and .tif files """
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"Dataset {dataset_name} not found in configuration registry.")

    config = DATASETS_CONFIG[dataset_name]
    img_path = os.path.join(dataset_dir, config['img'])
    gt_path = os.path.join(dataset_dir, config['gt'])

    # Load Image and Ground Truth based on file extension
    if config['img'].endswith('.mat'):
        image = io.loadmat(img_path)[config['img_key']]
        gt = io.loadmat(gt_path)[config['gt_key']]
    elif config['img'].endswith('.tif') or config['img'].endswith('.tiff'):
        image = tiff.imread(img_path)
        gt = tiff.imread(gt_path)
        # Fix dimension order if TIF is Channel-First (Bands, H, W) -> (H, W, Bands)
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = image.transpose((1, 2, 0))
    else:
        raise TypeError("Unsupported file format. Use .mat or .tif")

    # Preprocessing: Filter NaN values
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # Normalization (Standardization method)
    image = np.asarray(image, dtype=np.float32)
    # Scaled to [0, 1]
    denom = np.max(image) - np.min(image)
    if denom == 0: denom = 1e-8
    image = (image - np.min(image)) / denom
    
    # Zero-centering by spectral bands
    mean_by_c = np.mean(image, axis=(0, 1))
    for c in range(image.shape[-1]):
        image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # Handle Labels
    # Convert labels to start from 0 for the model, and -1 for undefined/background
    gt = gt.astype('int') - 1
    
    # Generate labels list dynamically (skipping -1)
    unique_labels = np.unique(gt)
    labels = [f"Class {int(i)}" for i in unique_labels if i >= 0]

    return image, gt, labels

def sample_gt(gt, percentage, seed):
    """ Split Ground Truth into training and testing masks """
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    # Stratified split ensures all classes are represented in train/test
    train_indices, test_indices = sklearn.model_selection.train_test_split(
        X, train_size=percentage, random_state=seed, stratify=y
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    return train_gt, test_gt

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True):
        """
        PyTorch Dataset for HSI patch extraction.
        :param image: 3D numpy array (H, W, Bands)
        :param gt: 2D numpy array (H, W) where -1 is ignored
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2 
        
        # Reflect padding allows pixels at the very edge to be the center of a patch
        self.data = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        # Find all valid pixels (where label >= 0)
        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        # Create coordinate list for all valid center pixels
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image.shape[0] + self.ps
                                 and self.ps <= y < image.shape[1] + self.ps])
        np.random.shuffle(self.indices)

    def hsi_augment(self, data):
        """ Standard spatial augmentations """
        if np.random.random() > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2: data = np.fliplr(data)
            elif 0.2 < prob <= 0.4: data = np.flipud(data)
            elif 0.4 < prob <= 0.6: data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8: data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0: data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.ps, y - self.ps
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # Extract the spatial-spectral patch
        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            data = self.hsi_augment(data)

        # Convert to PyTorch format (Channels, Height, Width)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Add 4th dimension (C, H, W) -> (1, C, H, W) to match model expectations
        return torch.from_numpy(data).unsqueeze(0), torch.from_numpy(label)
