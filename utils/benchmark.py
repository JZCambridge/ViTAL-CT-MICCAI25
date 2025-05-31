import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import os
import sys

if __name__ == "__main__":
    # Add the parent directory of src to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Custom utilities
    from image import *
    import dataset as ds
else:
    from utils.image import *
    import utils.dataset as ds

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.root = root
        self.transform = transform
        self.filenames = self.get_filenames()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label_ori = self.dataset[idx]
        label = F.one_hot(torch.tensor(label_ori), num_classes=2).float()
        filepath = self.dataset.imgs[idx][0]  # Get the file path
        return image, label, filepath
    
    def get_label(self, filename):
        idx = self.filenames.index(filename)
        return F.one_hot(torch.tensor(self.dataset[idx][1]), num_classes=2).float()
    
    def get_filenames(self):
        return [self.dataset.imgs[idx][0] for idx in range(len(self.dataset))]


def pneumonia_data(path, val_per=0.2, batch=32, date_setting='raw'):
    # dir
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')

    # transform
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    # datasets
    train_dataset = CustomImageFolder(train_dir, transform=data_transforms)

    train_size = int((1-val_per) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch, shuffle=True)

    test_dataset = CustomImageFolder(test_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # setting weights
    if date_setting == 'raw':
        weights = torch.tensor([1, 1])
    elif date_setting == 'weighted':
        # weights 
        weights, labels = ds.stratification_weight_generate(filenames=train_dataset.filenames, get_label_func=train_dataset.get_label, label='HRP2', debug=True, weight_type='bce')

        # check labels are correct (debug)
        # check_labels(train_dataset.filenames, labels)
        
    elif date_setting == 'balanced': # setting balanced
        _, labels = ds.stratification_weight_generate(filenames=train_dataset.filenames, get_label_func=train_dataset.get_label, label='HRP2', debug=True, weight_type='bce')
        
        # balance dataset
        train_idx = np.arange(len(train_dataset))
        train_idx, positive_values = ds.balance_dataset(train_idx, labels, label_type='HRP2', debug=True)
        weights = ds.weight_generate(positive_values, debug=True) # update weights should be approx equal
        ## update train_loader
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)

    # # check balance of dataset
    # check_balance(train_dataset.filenames, labels, train_idx)

    return train_loader, val_loader, test_loader, weights

def pneumonia_data_CV(path, val_per=0.2, batch=32, data_setting='raw'):
    # dir
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')

    # transform
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    # datasets
    train_dataset = CustomImageFolder(train_dir, transform=data_transforms)

    train_size = int((1-val_per) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch, shuffle=True)

    test_dataset = CustomImageFolder(test_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # weights 
    weights_tmp, labels = ds.stratification_weight_generate(filenames=train_dataset.filenames, get_label_func=train_dataset.get_label, label='HRP2', debug=True, weight_type='bce')

    # check labels are correct (debug)
    # check_labels(train_dataset.filenames, labels)
    
    # setting balanced
    if data_setting == 'raw':
        weights = torch.tensor([1, 1])
        balance = False
    elif data_setting == 'weighted':
        weights = weights_tmp
        balance = False
    else:
        weights = torch.tensor([1, 1])
        balance = True

    # stratification & CV & balance
    train_idx = np.arange(len(train_dataset))
    fold_idx_dict, weights = ds.stratified_k_fold_split_and_balance(
        train_idx, 
        labels, 
        fold=5, 
        balance=balance, 
        in_weights=weights,
        label='HRP2', 
        debug=True,)
    
    # check CV
    # print(f'Checking SKF split and balance\n')
    # for fold in range(5):
    #     print(f'Fold {fold}')
    #     print('Train')
    #     check_balance(train_dataset.filenames, labels, fold_idx_dict[f'train_{fold}'])
    #     print('Val')
    #     check_balance(train_dataset.filenames, labels, fold_idx_dict[f'val_{fold}'])
    # print(f'weights: {weights}')

    # update train_loaders and val_loaders
    train_loaders = []
    val_loaders = []

    for fold in range(5):
        train_subset = torch.utils.data.Subset(train_dataset, fold_idx_dict[f'train_{fold}'])
        val_subset = torch.utils.data.Subset(train_dataset, fold_idx_dict[f'val_{fold}'])
        train_loaders.append(DataLoader(train_subset, batch_size=batch, shuffle=True))
        val_loaders.append(DataLoader(val_subset, batch_size=batch, shuffle=False))     

    return train_loaders, val_loaders, test_loader, weights

def check_labels(filenames, labels):
    name_labels = np.zeros(len(filenames))

    for i, filename in enumerate(filenames):
        name_labels[i] = 1 * ('PNEUMONIA' in filename)
    
    print(f'Checking labels of dataset\n'
        f'Number of PNEUMONIA cases: {np.sum(name_labels)} \n'
          f'Number of NORMAL cases: {len(name_labels) - np.sum(name_labels)}\n',
          f'Incorrect labels: {np.sum(1 * ((name_labels - labels) != 0))}')

def check_balance(filenames, labels, train_idx):
    print(f'Checking balance of dataset\n'
          f'train_idx: {len(train_idx)}\n'
          f'Type of train_idx: {type(train_idx)}\n')
    
    filenames_balanced = [filenames[i] for i in train_idx]
    labels_balanced = [labels[i] for i in train_idx]
    name_labels = np.zeros(len(train_idx))

    for i, filename in enumerate(filenames_balanced):
        name_labels[i] = 1 * ('PNEUMONIA' in filename)
    
    print(f'Checking balance of dataset\n'
        f'Number of PNEUMONIA cases: {np.sum(name_labels)} \n'
          f'Number of NORMAL cases: {len(name_labels) - np.sum(name_labels)}\n',
          f'Incorrect labels: {np.sum(1 * ((name_labels - labels_balanced) != 0))}')

if __name__ == '__main__':
    train_loader, val_loader, test_loader, weights = pneumonia_data('/home/jay/Documents/chest_pneumonia_xray/chest_xray')
    print(len(train_loader), len(val_loader), len(test_loader))
    
    # # Print first ten train_loader cases labels
    # for i, (images, labels) in enumerate(train_loader):
    #     if i >= 10:
    #         break
    #     print(f"Batch {i+1} labels: {labels}")



