import sys
import os
import torch
import nibabel as nib
import copy
from nibabel.imageglobals import LoggingOutputSuppressor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import logging

# pipline
from torchvision import transforms # torch v1 pipline
from torchvision.transforms import InterpolationMode
from concurrent.futures import ThreadPoolExecutor

import time

if __name__ == "__main__":
    print('Running as a script')
    # Add the parent directory of src to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Custom utilities
    from image import *
    import dataset as ds
    import config as config
    # from utils.image import *
    # import utils.dataset as ds
    # import utils.config as config
else:
    from utils.image import *
    import utils.dataset as ds
    import utils.config as config

# global
sqrt_slices = 14 # 224 / patch_szie
patch_size = 16

# supress error log in: pixdim[0] (qfac) should be 1 (default) or -1; setting qfac to 1 (https://github.com/incf-nidash/nidmresults-fsl/issues/52)
# Set logging level to suppress info messages from nibabel
logging.getLogger('nibabel').setLevel(logging.ERROR)

def Dataloader_CTA(cfg, 
                    filepath, 
                    table_path, 
                    label='HRP5', 
                    train_proportion=0.75, 
                    val_proportion=0.15, 
                    tensor_output=False, 
                    slices_thres_dict=None, 
                    split='stratified', 
                    balance=True,
                    debug=False, 
                    input_data_type='tenosr',
                    debug_size=640):
    """
    Creates DataLoader objects for training, validation, and testing datasets.
    
    Parameters:
    - cfg: Configuration object containing parameters like batch size, number of workers, and loss function.
    - filepath: Path to the directory containing the dataset files.
    - table_path: Path to the table containing metadata for the dataset.
    - label: The target label used for training.
    - train_proportion: Proportion of the dataset to use for training.
    - val_proportion: Proportion of the dataset to use for validation.
    - tensor_output: Whether to output data as tensors.
    - slices_thres_dict: Dictionary specifying threshold values for slices per label.
    - split: Method for splitting the dataset ('random' or 'stratified').
    - debug: Boolean flag for enabling debug mode.
    - debug_size: Number of samples to use in debug mode.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    - weight: Weights for the stratified sampling.
    """

    if slices_thres_dict is None:
        slices_thres_dict = {'RCA': 20, 'LAD': 20, 'LCX': 20}

    # Initialize dataset
    if input_data_type == 'nifti':
        dataset = NiftiDataset(
            directory=filepath, 
            table_path=table_path, 
            label=label, 
            tensor_output=tensor_output, 
            slices_thres_dict=slices_thres_dict,
            random_settings=cfg.random_settings
        )
    elif input_data_type == 'tensor':
        dataset = TensorDataset_Combined(
            directory=filepath, 
            table_path=table_path, 
            postion=True,
            unet=True,
            label=label)
    elif input_data_type == 'tensor_aug':
        train_path = os.path.join(filepath, 'train')
        dataset = TensorDataset(
            directory=train_path, 
            table_path=table_path, 
            label=label)
        
        val_path = os.path.join(filepath, 'val')
        val_dataset = TensorDataset(
            directory=val_path, 
            table_path=table_path, 
            label=label)
        
        test_path = os.path.join(filepath, 'test')
        test_dataset = TensorDataset(
            directory=test_path, 
            table_path=table_path, 
            label=label)
    elif input_data_type == 'tensor_combined':
        train_path = os.path.join(filepath, 'train')
        dataset = TensorDataset_Combined(
            directory=train_path, 
            table_path=table_path, 
            postion=True,
            unet=True,
            label=label)
        
        val_path = os.path.join(filepath, 'val')
        val_dataset = TensorDataset_Combined(
            directory=val_path, 
            table_path=table_path, 
            postion=True,
            unet=True,
            label=label)
        
        test_path = os.path.join(filepath, 'test')
        test_dataset = TensorDataset_Combined(
            directory=test_path, 
            table_path=table_path, 
            postion=True,
            unet=True,
            label=label)
        
    elif input_data_type == 'tensor_aug_layer':
        train_path = os.path.join(filepath, 'train')
        dataset = TensorDataset_Reslayer(
            directory=train_path, 
            table_path=table_path, 
            label=label)
        
        val_path = os.path.join(filepath, 'val')
        val_dataset = TensorDataset_Reslayer(
            directory=val_path, 
            table_path=table_path, 
            label=label)
        
        test_path = os.path.join(filepath, 'test')
        test_dataset = TensorDataset_Reslayer(
            directory=test_path, 
            table_path=table_path,
            label=label)
    
    else:
        raise ValueError(f"Invalid input data type '{input_data_type}'. Use 'nifti' or 'tensor'.")

    # Generate stratification weight
    weights, labels = ds.stratification_weight_generate(filenames=dataset.filenames, get_label_func=dataset.get_label, label=label, debug=debug, weight_type=cfg.loss_func)
    
    # Split dataset
    if debug and debug_size != 0:
        logging.info("=================Debugging====================")
        train_size = int(debug_size * train_proportion)
        val_size = int(debug_size * val_proportion)
        test_size = debug_size - train_size - val_size
        train_dataset, _temp_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        # Disable augmentation for validation and test datasets
        if isinstance(_temp_dataset, torch.utils.data.Subset):
            _temp_dataset.dataset.random_settings['augmentation'] = False
        
        val_dataset, _temp_dataset = random_split(_temp_dataset, [val_size, len(_temp_dataset) - val_size])
        test_dataset, _remaining_dataset = random_split(_temp_dataset, [test_size, len(_temp_dataset) - test_size])
    else:
        split_methods = {
            'random': ds.random_split_dataset,
            'stratified': ds.stratified_split_dataset
        }

        if split in split_methods:
            if split == 'stratified':
                    # Split dataset: internal and external split
                    internal_idx, external_idx = train_test_split(
                        np.arange(len(dataset)),
                        test_size=val_proportion,
                        stratify=labels,
                        random_state=config.RANDOM_STATE
                    )

                    # external dataset and loader
                    test_dataset = Subset(dataset, external_idx)

                    # Split dataset: internal split
                    train_idx_tmp, val_idx = train_test_split(
                        internal_idx,
                        test_size=val_proportion,
                        stratify=labels[internal_idx],
                        random_state=config.RANDOM_STATE
                    )

                    # external dataset and loader
                    val_dataset = Subset(dataset, val_idx)

                    train_idx_tmp_new, positive_values = ds.balance_dataset(np.arange(len(train_idx_tmp)), labels[train_idx_tmp], label_type=label, debug=True)
                    train_idx = [train_idx_tmp[i] for i in train_idx_tmp_new]
                    train_dataset = Subset(dataset, train_idx)

                    weights = ds.weight_generate(positive_values, debug=True) # update weights should be approx equal
            
            elif split == 'random':
                train_dataset, val_dataset, test_dataset = split_methods[split](
                    dataset, train_proportion, val_proportion
                )
        elif split == 'none':
            if balance:
                train_idx, positive_values = ds.balance_dataset(np.arange(len(dataset)), labels, label_type=label, debug=True)
                train_dataset = Subset(dataset, train_idx)
                weights = ds.weight_generate(positive_values, debug=True)
            pass        
        else:
            raise ValueError(f"Invalid split type '{split}'. Use 'random' or 'stratified'.")

    # DataLoaders settings for training
    train_loader_params = {
        'batch_size': cfg.bs,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'drop_last': False
    }

    # DataLoaders settings for validation and test
    val_test_loader_params = {
        'batch_size': cfg.bs,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'drop_last': False
    }

    # remove augmentation for validation and test datasets
    if input_data_type == 'nifti':
        val_dataset.random_settings['normal_augmentation'] = False
        val_dataset.random_settings['torch_augmentation'] = False
        test_dataset.random_settings['normal_augmentation'] = False
        test_dataset.random_settings['torch_augmentation'] = False

    # create dataloaders
    train_loader = DataLoader(train_dataset, **train_loader_params)
    val_loader = DataLoader(val_dataset, **val_test_loader_params)
    test_loader = DataLoader(test_dataset, **val_test_loader_params)

    return train_loader, val_loader, test_loader, weights

def Dataloader_CTA_CV(cfg, 
                    filepath, 
                    table_path, 
                    label='HRP5', 
                    external_proportion=0.15, 
                    fold=5,
                    tensor_output=True, 
                    slices_thres_dict=None, 
                    balance=True,
                    debug=False, 
                    input_data_type='tenosr',
                    split='stratified'):
    """
    Creates DataLoader objects for training, validation, and testing datasets.
    
    Parameters:
    - cfg: Configuration object containing parameters like batch size, number of workers, and loss function.
    - filepath: Path to the directory containing the dataset files.
    - table_path: Path to the table containing metadata for the dataset.
    - label: The target label used for training.
    - external_proportion: Proportion of the dataset to use for the external (test) split.
    - fold: Number of folds for Stratified K-Fold cross-validation.
    - tensor_output: Whether to output data as tensors.
    - slices_thres_dict: Dictionary specifying threshold values for slices per label.
    - split: Method for splitting the dataset ('random' or 'stratified').
    - balance: Boolean flag to balance the training data in each fold.
    - debug: Boolean flag for enabling debug mode.
    - debug_size: Number of samples to use in debug mode.

    Returns:
    - train_loaders: DataLoader for the training set.
    - val_loaders: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    - weights: Weights for the stratified sampling.
    """

    if slices_thres_dict is None:
        slices_thres_dict = {'RCA': 20, 'LAD': 20, 'LCX': 20}

    # DataLoaders settings for training
    train_loader_params = {
        'batch_size': cfg.bs,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'drop_last': False
    }

    # DataLoaders settings for validation and test
    val_test_loader_params = {
        'batch_size': cfg.bs,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'drop_last': False
    }

    # Initialize dataset
    if input_data_type == 'nifti':
        dataset = NiftiDataset(
            directory=filepath, 
            table_path=table_path, 
            label=label, 
            tensor_output=tensor_output, 
            slices_thres_dict=slices_thres_dict,
            random_settings=cfg.random_settings
        )
    elif input_data_type == 'tensor':
        dataset = TensorDataset(
            directory=filepath, 
            table_path=table_path, 
            label=label)

    # Generate stratification weight
    weights, labels = ds.stratification_weight_generate(filenames=dataset.filenames, get_label_func=dataset.get_label, label=label, debug=debug, weight_type=cfg.loss_func)

    if debug:
        # get unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        print('=== CTA CV Debugging... ===')
        print(f'*Initial*')
        print(f"Total labels length: {len(labels)}")
        print(f"Total Dataset length: {len(dataset)}")
        print(f"Unique labels: {unique_labels}")
        print(f"Label counts: {counts}")
        print(f'Ratio: {counts[0] / counts[1]:.2f}')

    # Split dataset: internal and external split
    internal_idx, external_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=external_proportion,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )

    # external dataset and loader
    external_dataset = Subset(dataset, external_idx)
    # remove augmentation for external dataset
    if input_data_type == 'nifti':
        external_dataset.random_settings['normal_augmentation'] = False
        external_dataset.random_settings['torch_augmentation'] = False
    test_loader = DataLoader(external_dataset, **val_test_loader_params)

    if debug:
        # internal and external labels and counts
        internal_labels = [labels[i] for i in internal_idx]
        external_labels = [labels[i] for i in external_idx]
        internal_unique_labels, internal_counts = np.unique(internal_labels, return_counts=True)
        external_unique_labels, external_counts = np.unique(external_labels, return_counts=True)
        print(f'Initial internal indices length: {len(internal_idx)}')
        print(f'Intrnal unique labels: {internal_unique_labels}')
        print(f'Internal label counts: {internal_counts}')
        print(f'Ratio: {internal_counts[0] / internal_counts[1]:.2f}')
        print(f'Initial external indices length: {len(external_idx)}')
        print(f'External unique labels: {external_unique_labels}')
        print(f'External label counts: {external_counts}')
        print(f'Ratio: {external_counts[0] / external_counts[1]:.2f}')

    # Stratified K-Fold split and balance using the new function
    fold_idx_dict, updated_weights = ds.stratified_k_fold_split_and_balance(
        internal_idx=internal_idx,
        internal_labels=[labels[i] for i in internal_idx],
        fold=fold,
        balance=balance,
        in_weights=weights,
        label=label,
        debug=debug
    )

    # internal datasets and loaders
    train_loaders = {}
    val_loaders = {}

    if debug:
        print(f'After Stratified K-Fold split and balance...')

    for i in range(fold):
        if debug:
            print(f'fold: {i}')

        # fold training and validation indices
        training_idx = fold_idx_dict[f'train_{i}']
        val_idx = fold_idx_dict[f'val_{i}']

        if debug:
            # training and validation labels and counts
            training_labels = [labels[i] for i in training_idx]
            val_labels = [labels[i] for i in val_idx]
            training_unique_labels, training_counts = np.unique(training_labels, return_counts=True)
            val_unique_labels, val_counts = np.unique(val_labels, return_counts=True)
            print(f'training_idx length: {len(training_idx)}')
            print(f'train unique labels: {training_unique_labels}')
            print(f'training label counts: {training_counts}')
            print(f'Ratio: {training_counts[0] / training_counts[1]:.2f}')
            print(f'validation_idx length: {len(val_idx)}')
            print(f'val unique labels: {val_unique_labels}')
            print(f'val label counts: {val_counts}')
            print(f'Ratio: {val_counts[0] / val_counts[1]:.2f}')

        # Create train dataset & loader
        train_dataset = Subset(dataset, training_idx)
        train_loaders[i] = DataLoader(train_dataset, **train_loader_params)

        # Create validation dataset & loader
        val_dataset = Subset(dataset, val_idx)
        # remove augmentation for validation dataset
        if input_data_type == 'nifti':
            val_dataset.random_settings['normal_augmentation'] = False
            val_dataset.random_settings['torch_augmentation'] = False
        val_loaders[i] = DataLoader(val_dataset, **val_test_loader_params)

    return train_loaders, val_loaders, test_loader, updated_weights

class NiftiDataset(Dataset):
    def __init__(self, 
                 directory=None, 
                 table_path=None, 
                 label="HRPs", 
                 tensor_output=False, 
                 display=False, 
                 result_list=['posremod', 'lowatten', 'napkin', 'spotty', 'punctate'], 
                 slices_thres_dict={'RCA': 20, 'LAD': 20, 'LCX': 20}, 
                 debug=False, 
                 random_settings=None,
                 label_calculation=False,
                 slice_thresholding=False):
        super(NiftiDataset, self).__init__()
        self.directory = directory
        self.directories, self.filenames = ds.get_files(directory, end=".nii.gz")
        self.df = pd.read_csv(table_path)
        self.display = display
        self.label = label
        self.tensor_output = tensor_output
        self.result_list = result_list
        self.slices_thres_dict = slices_thres_dict
        self.debug = debug
        self.label_calculation = label_calculation
        self.total_files = None
        self.positive_values = None

        # Initialize random settings with default values if not provided
        self.random_settings = random_settings or {
            'normal_augmentation': False, 'torch_augmentation': False,
            'column_flip': False, 'column_flip_settings': {'random_flip': True, 'flip_height': True, 'flip_width': True, 'flip_diagonal': True, 'flip_antidiagonal': True},
            'column_rotate': False, 'angle_range': 90,
            'column_shift': False, 'random_shift': True, 'shift_range': 4,
            'frame_swing': False, 'frame_swing_range': 4,
            'white_noise_2D': False, 'noise_scale': 40,
            'gaussian_filter_2D': False, 'gaussian_random': True, 'sigma': 1.2,
            'mask': False, 'mask_portion': 0.5
        }

        # dataframe Replace NaN values with 0
        self.df.fillna(0, inplace=True)

        # Apply slices thresholding
        if slice_thresholding:
            self.filenames, self.directories = ds.slices_thresholding(self.filenames, self.directories, self.df, self.slices_thres_dict, self.debug)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load the NIfTI file
        filepath = self.filenames[idx]
        nifti_img = nib.load(filepath)
        data_array = nifti_img.get_fdata()

        # Apply augmentations
        data_array = self.apply_augmentations(data_array)

        # Output type
        output_image = self.process_output(data_array)

        # Return label
        output_label = self.get_label(filepath)

        # Display if required
        if self.display:
            self.display_output(data_array, output_image)

        return output_image, output_label, filepath
    
    def apply_augmentations(self, data_array):
        """Applies augmentations to the input data array."""
        if self.random_settings['normal_augmentation']:
            if self.random_settings['column_shift']:
                data_array = shift_array_along_axis(data_array, random=self.random_settings['random_shift'], shift_1st=self.random_settings['shift_range'], shift_2nd=self.random_settings['shift_range'])
            if self.random_settings['column_flip']:
                data_array = flip_height_width_diagonals(data_array, setting=self.random_settings['column_flip_settings'])
            if self.random_settings['column_rotate']:
                data_array = rotate_along_depth_random_angle(data_array, angle_range=self.random_settings['angle_range'])
            if self.random_settings['white_noise_2D']:
                data_array = add_integer_white_noise_to_slices_range(data_array, noise_range=self.random_settings['noise_scale'])
            if self.random_settings['gaussian_filter_2D']:
                data_array = apply_gaussian_filter_to_slices(data_array, random=self.random_settings['gaussian_random'], sigma=self.random_settings['sigma'])
        return data_array 

    def process_output(self, data_array):
        """Processes the output as either tensor or image."""
        if self.tensor_output:
            data_array = np.array(data_array, dtype=np.float32)
            tensor = torch.tensor(data_array, dtype=torch.float32)
            output_tensor = self.process_tensor_output(tensor)
            return output_tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            image = Image.fromarray(data_array[:, :, 0].astype(np.float32))
            centered_image = self.take_center(image, patch_size=patch_size, series_interval=5, random_range=self.random_settings['frame_swing_range'], frame_swing=self.random_settings['frame_swing'], tensor=self.tensor_output)
            output_image_1C = self.concatenate_slices_img(data=data_array, patch_size=patch_size, image_size=(sqrt_slices * patch_size, sqrt_slices * patch_size), series_interval=5, debug=True)
            return output_image_1C.convert("RGB")

    def process_tensor_output(self, tensor):
        """Applies augmentations or centering to tensor output."""
        if self.random_settings['torch_augmentation']:
            centered_tensor = self.take_center_augmentation(tensor, patch_size=patch_size, series_interval=5, random_range=self.random_settings['frame_swing_range'], tensor=self.tensor_output)
            return self.concatenate_augmentation(self.concatenate_slices_tensor(centered_tensor), tensor=True)
        else:
            centered_tensor = self.take_center(tensor, patch_size=patch_size, series_interval=5, random_range=0, frame_swing=self.random_settings['frame_swing'], tensor=self.tensor_output)
            return self.concatenate_slices_tensor(centered_tensor, mask=self.random_settings['mask'], portion=self.random_settings['mask_portion'],)        

    def get_label(self, filepath):
        if self.label_calculation:
            return ds.get_label_calculation(df=self.df, filepath=filepath, driectory=self.directories[idx], label=self.label, result_list=self.result_list, debug=self.debug)
        else:
            return ds.get_label_from_df(self.df, filepath, self.label)

    @staticmethod
    def take_center(img, patch_size=patch_size, series_interval=5, random_range=0, frame_swing=True, tensor=False):
        """
        Extract a 'patch_size'x'patch_size' region 
        and pick every 'series_interval' slices 
        around the center of the tensor.
        """
        # Get the center of the image
        if tensor: 
            center_x, center_y, slices_z = img.shape[0] // 2, img.shape[1] // 2, img.shape[2]
        else: 
            width, height, slices = img.size
            center_x, center_y = width // 2, height // 2
        
        # Extract a patch region around the center
        start_x = center_x - patch_size // 2
        end_x = center_x + patch_size // 2
        start_y = center_y - patch_size // 2
        end_y = center_y + patch_size // 2

        # random center range
        if random_range == 0 or not frame_swing:
            if tensor: 
                centered_img = img[start_x:end_x, start_y:end_y, ::series_interval]
            else:
                centered_img = img.crop((start_y, start_x, end_y, end_x))
        
        else:
            if tensor: 
                # Initialize an empty list to store slices
                slices = []

                # Loop through the third dimension with the series interval
                for i in range(0, slices_z, series_interval):
                    # Generate a random offset
                    x_offset = np.random.randint(-abs(random_range+1), abs(random_range+1))
                    y_offset = np.random.randint(-abs(random_range+1), abs(random_range+1))

                    # Extract a patch region around the center
                    start_x_offset = start_x + x_offset
                    end_x_offset = end_x + x_offset
                    start_y_offset = start_y + y_offset
                    end_y_offset = end_y + y_offset
                    
                    # Crop the image slice with the adjusted start_x
                    slice_img = img[start_x_offset:end_x_offset, start_y_offset:end_y_offset, i]
                    
                    # Append the slice to the list
                    slices.append(slice_img)

                # Combine the slices back into a single image tensor (if needed)
                centered_img = torch.stack(slices, dim=2)

        return centered_img
    
    @staticmethod
    def take_center_augmentation(img, patch_size=patch_size, series_interval=5, random_range=0, tensor=False, debug=False):
        """
        Extract a 'patch_size'x'patch_size' region 
        and pick every 'series_interval' slices 
        around the center of the tensor.
        """
        # Get image size
        if tensor: 
            center_x, center_y, slices_z = img.shape[0] // 2, img.shape[1] // 2, img.shape[2]
        else: 
            width, height, slices = img.size
            center_x, center_y = width // 2, height // 2

        if tensor: 
            # Initialize an empty list to store slices
            slices = []

            # timing
            start = time.time()

            transform = transforms.Compose([
                transforms.CenterCrop((patch_size + abs(random_range), patch_size + abs(random_range))), # Center crop with random range
                transforms.RandomApply([transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.8, 1), shear=(2, 2, 2, 2), interpolation=InterpolationMode.BILINEAR)], p=0.5), # Random affine transformation
                transforms.RandomCrop(size=(patch_size, patch_size), padding=2, pad_if_needed=True), # Random crop
                transforms.RandomHorizontalFlip(p=0.5), # Random horizontal flip
                transforms.RandomVerticalFlip(p=0.5), # Random vertical flip
                #transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2))], p=0.5), # Gaussian blur
            ])

            for i in range(0, slices_z, series_interval):
                slice_img = img[:, :, i]
                if slice_img.ndim == 2:  # Ensure the slice is 3D (C, H, W)
                    slice_img = slice_img.unsqueeze(0) # Add a channel dimension
                transformed_slice = transform(slice_img)
                slices.append(transformed_slice[0, :, :])

            centered_img = torch.stack(slices, dim=2)

            # timing
            end = time.time()
            if debug: print(f'centered_img torch time: {end - start}')

        return centered_img
    
    @staticmethod
    def concatenate_augmentation(img, tensor=False, debug=False):
        """
        """
        # Get image size
        if tensor: 
            # Ensure the input tensor has the correct shape
            if img.ndim == 2:  # If the input is 2D (H, W), add a channel dimension
                img_use = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[0] != 1:  # If the input is 3D but not (1, H, W), add a batch dimension
                img_use = img.unsqueeze(0)
            
            # timing
            start = time.time()

            transform3D = transforms.Compose([
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2))], p=0.5), # Gaussian blur
                transforms.RandomInvert(p=0.5), # Random invert
                transforms.RandomSolarize(threshold=300, p=0.5), # Random solarize
                transforms.RandomAdjustSharpness(2, p=0.5), # Random sharpness
                transforms.RandomAutocontrast(p=0.5), # Random autocontrast
            ])

            transform = transforms.RandomApply([transform3D], p=0.5)

            augmented_img = transform(img_use)

            # timing
            end = time.time()
            if debug: print(f'augmented_img torch time: {end - start}')

        return augmented_img[0, :, :]

    @staticmethod
    def concatenate_slices_tensor(img, image_size=(sqrt_slices*patch_size, sqrt_slices*patch_size), patch_size=patch_size, mask=False, portion=0.5, debug=False):
        """
        Concatenate the slices to form an image of (512, 512) by stacking the slices
        """
        # flip tensor (D1, D2, D3) to (D3, D1, D2)
        tensor = img.permute(2, 0, 1)

        w, h = image_size[0] // patch_size, image_size[1] // patch_size
        total_slices_needed = w * h

        # Padding with black slices
        slices_to_pad = total_slices_needed - tensor.size(0)
        if slices_to_pad > 0:
            padding = torch.zeros(slices_to_pad, patch_size, patch_size)
            tensor = torch.cat([tensor, padding], dim=0)
        
        # Masking
        if mask:
            # generate random numbers between 0 and 1 with lenth of tensor.size(0)
            random_numbers = torch.rand(tensor.size(0))
            # get the index of the random numbers greater than portion
            mask_index = torch.where(random_numbers < portion)
            # set the masked slices to 0
            tensor[mask_index] = 0

        # Concatenate slices to form a 512x512 image
        output_img = torch.cat(
            [
                torch.cat([tensor[i * h + j] for j in range(w)], dim=1)
                for i in range(h)
            ],
            dim=0,
        )

        assert output_img.shape == image_size

        return output_img
    
    @staticmethod
    def img_transform(data:np.array, thresdict = {'lower':-100, 'upper':150}):
        '''
        # clipping upper lower and linear transform [0, 255]
        '''
        
        # Step 1: Apply the clipping
        data_array = np.clip(data, thresdict['lower'], thresdict['upper'])

        # Step 2: Normalize the remaining values to the range 0 to 255
        # First, shift the values to be non-negative
        data_array = data_array - np.min(data_array)

        # Then scale to the range 0 to 255
        data_array = (data_array / np.max(data_array)) * 255

        # Step 3: Convert to integer form
        data_array = data_array.astype(np.uint8)

        return data_array

    def concatenate_slices_img(self, data, patch_size=patch_size, image_size=(sqrt_slices*patch_size, sqrt_slices*patch_size), series_interval=5, debug=True):
        # stack centered slices
        slices = []
        if debug: 
            print(f'data.shape[2]: {data.shape[2]}')
        for i in range(0, data.shape[2], series_interval):
            # Convert to a PIL image
            image = Image.fromarray(data[:,:,i].astype(np.float32))
            centered_image = self.take_center(image, patch_size=patch_size, series_interval=5, tensor=False, random_range=self.random_range)
            slices.append(centered_image)
        
        if debug: 
            print(f'slices len: {len(slices)}')
        # stacking
        # Create a new blank image to paste the slices into
        output_img = Image.new("F", image_size)
        
        # Calculate the number of slices per row
        slices_per_row = image_size[0] // patch_size

        # Paste each slice into the output image at the correct location
        for i, slice in enumerate(slices):
            y_offset = (i // slices_per_row) * patch_size
            x_offset = (i % slices_per_row) * patch_size
            output_img.paste(slice, (x_offset, y_offset))

        return output_img

    def display_output(self, data_array, output_image):
        if self.tensor_output:
            self.display_tensor(torch.tensor(data_array[:, :, 0]), "Original Image")
            self.display_tensor(output_image[0, :, :], "Concatenated Image")
        else:
            self.display_image(Image.fromarray(data_array[:, :, 0]), "Original Image")
            self.display_image(output_image, "Concatenated Image")

    def display_tensor(self, tensor, title=""):
        """
        Display the tensor as an image
        """
        plt.imshow(tensor, cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.title(title)
        plt.show()

    def display_image(self, image, title=""):
        """
        Display the image
        """
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(title)
        plt.show()


class TensorDataset(Dataset):
    def __init__(self, 
                 directory=None, 
                 table_path=None, 
                 display=False, 
                 label = "HRP2",
                 debug=False,):
        super(TensorDataset, self).__init__()
        self.directory = directory
        self.directories, self.filenames = ds.get_files(directory, end=".pt")
        self.df = pd.read_csv(table_path)
        self.display = display
        self.label = label
        self.debug = debug

        # dataframe Replace NaN values with 0
        self.df.fillna(0, inplace=True)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load the tensor file
        filepath = self.filenames[idx]
        output_image = torch.load(filepath)  # Load tensor
        # output_image = torch.load(filepath, weights_only=False)  # Load tensor for tan_gong
        #output_image = (output_image-torch.min(output_image))/(torch.max(output_image)-torch.min(output_image)+1e-5) ## add in 20250130
        # Return label
        output_label = self.get_label(filepath)

        # Display if required
        if self.display:
            self.display_tensor(output_image[0, :, :], "Concatenated Image")

        return output_image, output_label, filepath

    def get_label(self, filepath):
        return ds.get_label_from_df(self.df, filepath, self.label)

    def display_tensor(self, tensor, title=""):
        """
        Display the tensor as an image
        """
        plt.imshow(tensor, cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.title(title)
        plt.show()

class TensorDataset_Reslayer(Dataset):
    def __init__(self, 
                 directory=None, 
                 table_path=None, 
                 display=False, 
                 label = "HRP2",
                 postion = False,
                 debug=False,):
        super(TensorDataset_Reslayer, self).__init__()
        self.directory = directory
        self.directories, self.filenames = ds.get_files(directory, end=".pt")
        self.df = pd.read_csv(table_path)
        self.display = display
        self.label = label
        self.debug = debug
        self.postion = postion

        # dataframe Replace NaN values with 0
        self.df.fillna(0, inplace=True)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load the tensor file
        filepath = self.filenames[idx]
        output_image = torch.load(filepath)  # Load tensor
        #output_image = (output_image-torch.min(output_image))/(torch.max(output_image)-torch.min(output_image)+1e-5) ## add in 20250130
        # Return label
        output_label = self.get_label(filepath)
        output_layer = self.get_layer(filepath)

        if not self.postion:
            return output_image, output_label, filepath, output_layer
        else:
            out_position = self.get_position(filepath)
            return output_image, output_label, filepath, output_layer, out_position

    def get_label(self, filepath):
        return ds.get_label_from_df(self.df, filepath, self.label)
    
    def get_layer(self, filepath):
        return ds.get_res_layer(filepath, layer_path="/home/jay/Documents/VIT_CTA/Vit_CT_Repository/Output_Folder/features_resnet", layer_end="_features.pt", debug=self.debug)

    def get_position(self, filepath):
        return ds.get_center_position(filepath, folder_path = '/media/jay/Storage21/ScotHeart_Aug_ResVit/positions', file_end="_pos.pt", debug=False)

class TensorDataset_Combined(Dataset):
    def __init__(self, 
                 directory=None, 
                 table_path=None, 
                 display=False, 
                 label = "HRP2",
                 postion = False,
                 unet = False,
                 debug=False,):
        super(TensorDataset_Combined, self).__init__() ## revise
        self.directory = directory
        self.directories, self.filenames = ds.get_files(directory, end=".pt")
        self.df = pd.read_csv(table_path)
        self.display = display
        self.label = label
        self.debug = debug
        self.postion = postion
        self.unet = unet

        # dataframe Replace NaN values with 0
        self.df.fillna(0, inplace=True)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load the tensor file
        filepath = self.filenames[idx]
        output_image = torch.load(filepath)  # Load tensor
        # Return label
        output_label = self.get_label(filepath)

        # Return unet output
        if self.unet:
            output_unet = self.get_unet(filepath)
        else:
            output_unet = None

        # Return position
        if self.postion:
            out_position = self.get_position(filepath)
        else:
            out_position = None
        
        return output_image, output_label, filepath, out_position, output_unet

    def get_label(self, filepath):
        return ds.get_label_from_df(self.df, filepath, self.label)
    
    def get_unet(self, filepath):
        if os.path.exists(os.path.join('/media/jay/Storage21/ScotHeart_Aug_ResVit/unet',filepath)):
            output = ds.get_unet_info(filepath, folder_path = '/media/jay/Storage21/ScotHeart_Aug_ResVit/unet', file_end="_grid_features.pt", debug=False)
        else:
            output = torch.ones(196,256)
        return output
        #return ds.get_unet_info(filepath, folder_path = '/media/jay/Storage2/ScotHeart_Aug_ResVit/unet', file_end="_grid_features.pt", debug=False)

    def get_position(self, filepath):
        return ds.get_center_position(filepath, folder_path = '/media/jay/Storage21/ScotHeart_Aug_ResVit/positions', file_end="_pos.pt", debug=False)

if __name__ == "__main__":
    '''Test the NiftiDataset class'''
    # filepath = "/media/jay/Storage2/ScotHeart/GoodToshibaXsections/"
    # table_path = "/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP.csv"

    # dataset = NiftiDataset(filepath, 
    #                         table_path, 
    #                         label='HRP2', 
    #                         tensor_output=True, 
    #                         display=True, 
    #                         slices_thres_dict = {'RCA': 20, 'LAD':20, 'LCX':60}, 
    #                         debug=True,
    #                         random_settings={'normal_augmentation':True, 'torch_augmentation':False, 
    #                                     'column_flip':True, 'column_flip_settings':{'random_flip': True, 'flip_height': True, 'flip_width': True, 'flip_diagonal': True, 'flip_antidiagonal': True}, # Flip columns, False to disable
    #                                     'column_rotate':True, 'angle_range':90, # column-wise rotation, False to disable
    #                                     'column_shift':True, 'random_shift':True, 'shift_range':2, # Shift columns, False to disable
    #                                     'frame_swing': False, 'frame_swing_range': 4, # slice-wise shift, 0 to disable
    #                                     'white_noise_2D':False, 'noise_scale':40, # slice-wise noise, False to disable
    #                                     'gaussian_filter_2D':False, 'gaussian_random':True, 'sigma':1.2, # slice-wise gaussian, False to disabl
    #                                     },
    #                 label_calculation=False,
    #                 slice_thresholding=False)
    
    # print(len(dataset))

    # ds.stratification_weight_generate(filenames=dataset.filenames, get_label_func=dataset.get_label, label='HRP2', debug=True, weight_type='bce')

    # dataset[1]
    # dataset[2]
    # dataset[3]

    '''Test the TensorDataset class'''
    # # dataset = TensorDataset
    # filepath = "/media/jay/Storage2/ScotHeart_Aug/train"
    # table_path = "/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP.csv"

    # dataset = TensorDataset(filepath,
    #                         table_path,
    #                         display=True,
    #                         label='HRP2',
    #                         debug=True)

    # print(len(dataset))

    # for i in range(4):
    #     output_image, output_label, filepath = dataset[i]
    #     print(output_label, filepath)

    '''Test the TensorDataset_Reslayer class'''
    # dataset = TensorDataset
    filepath = "/media/jay/Storage21/ScotHeart_Aug_ResVit/train"
    table_path = "/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP.csv"

    dataset = TensorDataset_Combined(filepath,
                            table_path,
                            display=True,
                            label='HRP2',
                            postion=True,
                            unet=True,
                            debug=True)

    print(len(dataset))

    for i in range(4):
        output_image, output_label, filepath, out_position, output_unet = dataset[i]
        print(output_label, filepath)
        print(output_unet.shape)
        print(output_unet[0])



    # import argparse

    # # Command-Line Argument Parsing
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    # parser.add_argument("-fold", type=int, default=5, help="Fold number for cross-validation")
    # parser.add_argument("-data_path", type=str, default="/media/jay/Storage2/ScotHeart/GoodToshibaXsections/", help="Path to the data directory")
    # parser.add_argument("-data_info", type=str, default="HRP2", help="Data information type (e.g., HRP5, Ca3)")
    # parser.add_argument("-annotation", type=str, default="/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP.csv", help="Path to the annotation file, evnets:/home/jay/Documents/VIT_CTA/Vit_CT_Repository/0_Raw_Data/Toshiba_HRP_Events2018.csv")
    # parser.add_argument("-lr", type=float, default=1e-6, help="Learning rate") # TODO: fixed learning rate
    # parser.add_argument("-epochs", type=int, default=100, help="Number of training epochs")
    # parser.add_argument("-num_workers", type=int, default=3, help="Number of data loading workers")
    # parser.add_argument("-num_classes", "-nc", type=int, default=2, help="Number of classes")
    # # parser.add_argument("-backbone", type=str, default="vit_small_p16_224", help="Model backbone type")
    # # parser.add_argument("-backbone", type=str, default="vit_base_p16_224", help="Model backbone type")
    # parser.add_argument("-backbone", type=str, default="vit_base_p16_224", help="Model backbone type")
    # parser.add_argument("-train_type", "-tt", type=str, default="resnet", help="Training type: lora, full, resnet")
    # parser.add_argument("-rank", "-r", type=int, default=4, help="Rank for LoRA")
    # parser.add_argument("-alpha", "-a", type=int, default=4, help="Alpha for LoRA")
    # parser.add_argument("-loss_func", type=str, default='bce', help="Loss function type: bce, ce, ca3")
    # parser.add_argument("-ignore_index", type=int, default=2, help="Ignore index for cross-entropy loss")
    # #parser.add_argument("-weights_path", type=str, default='/home/jay/Documents/VIT_CTA/Vit_CT_Repository/Output_Folder/Run/20240709_001508.pt', help="Path to pre-trained weights file") # TODO: Changed path
    # parser.add_argument("-weights_path", type=str, default=None)
    # parser.add_argument("-performance_measure", type=str, default='loss', help="Performance measure: loss, auc")
    # parser.add_argument("-debug", type=bool, default=False, help="Debug mode")
    # parser.add_argument("-log_table_head", type=str, nargs='+', default=['posremod', 'lowatten', 'napkin', 'spotty', 'punctate'])
    # parser.add_argument("-split", type=str, default="stratified", help="stratified, random")
    # parser.add_argument("-balance", type=bool, default=True, help="Stratified sampling")
    # parser.add_argument("-debug_size", type=int, default=0, help="Debug size")
    # parser.add_argument("-check_incorrect", type=bool, default=False, help="Check all files and save incorrect files")
    # parser.add_argument("-benchmark", type=bool, default=False, help="Benchmark a know dataset")
    # parser.add_argument("-inbalance", type=str, default='balanced', help='data inbalance dealing method: raw, weighted, balanced')
    # parser.add_argument("-random_settings", type=dict, default={'normal_augmentation':True, 'torch_augmentation':False, 
    #                                     'column_flip':True, 'column_flip_settings':{'random_flip': True, 'flip_height': True, 'flip_width': True, 'flip_diagonal': True, 'flip_antidiagonal': True}, # Flip columns, False to disable
    #                                     'column_rotate':True, 'angle_range':90, # column-wise rotation, False to disable
    #                                     'column_shift':True, 'random_shift':True, 'shift_range':2, # Shift columns, False to disable
    #                                     'frame_swing': False, 'frame_swing_range': 4, # slice-wise shift, 0 to disable
    #                                     'white_noise_2D':False, 'noise_scale':40, # slice-wise noise, False to disable
    #                                     'gaussian_filter_2D':False, 'gaussian_random':True, 'sigma':1.2, # slice-wise gaussian, False to disabl
    #                                     }, help="Data augmentation")

    # cfg = parser.parse_args()


    # trainsets, valsets, testset, _ = Dataloader_CTA_CV(
    #     cfg, 
    #     filepath=cfg.data_path, 
    #     table_path=cfg.annotation,
    #     label=cfg.data_info, 
    #     external_proportion=0.15, 
    #     fold=cfg.fold,
    #     tensor_output=True,
    #     slices_thres_dict={'RCA': 20, 'LAD': 20, 'LCX': 20},
    #     balance=cfg.balance,
    #     split=cfg.split,
    #     debug=True,
    #     debug_size=0#133
    # )
