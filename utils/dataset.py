import os
import sys
import numpy as np
import ast
import re
import torch
from torch.utils.data import Dataset, random_split, Subset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Add the parent directory of src to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Custom utilities
    from .image import *
    import utils.config as config
else:
    from utils.image import *
    import utils.config as config
    # from image import *
    # import config as config


def random_split_dataset(dataset: Dataset, train_proportion: float, val_proportion: float):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    dataset (Dataset): The dataset to be split.
    train_proportion (float): The proportion of the dataset to include in the training set.
    val_proportion (float): The proportion of the dataset to include in the validation set.

    Returns:
    train_dataset (Dataset): The training subset of the dataset.
    val_dataset (Dataset): The validation subset of the dataset.
    test_dataset (Dataset): The test subset of the dataset.
    """
    train_size = int(len(dataset) * train_proportion)
    val_size = int(len(dataset) * val_proportion)
    test_size = len(dataset) - train_size - val_size

    train_dataset, temp_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    # Disable augmentation for validation and test datasets
    if isinstance(temp_dataset, torch.utils.data.Subset):
        temp_dataset.dataset.random_settings['augmentation'] = False

    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def stratified_split_dataset(dataset: Dataset, train_proportion: float, val_proportion: float, labels: np.array, balance=True, label_type='HRP5', debug=False):
    # Calculate test set proportion
    test_proportion = 1 - train_proportion - val_proportion

    # Initial stratified split into training and a temporary set
    train_temp_idx, temp_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=(val_proportion + test_proportion),
        stratify=labels,
        random_state=None # random split
    )

    # Convert train_temp_idx to a NumPy array for efficient operations
    train_idx = np.array(train_temp_idx)

    positive_values = None

    # balance the dataset
    if balance:
        train_labels = labels[train_idx]
        train_idx, positive_values = balance_dataset(train_idx, train_labels, label_type, debug)
    else:
        pass

    # Extract labels for the temporary set for further stratification
    temp_labels = [labels[i] for i in temp_idx]

    # Step 2: Stratified split of the temporary set into validation and test sets
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # Splitting the temporary set equally into validation and test sets
        stratify=temp_labels
    )
    print(f"train_idx: {train_idx}")

    if debug:
        # calculate the number of samples for each label for each set
        train_labels = [labels[i] for i in train_idx]
        train_temp_labels = [labels[i] for i in train_temp_idx]
        val_labels = [labels[i] for i in val_idx]
        test_labels = [labels[i] for i in test_idx]
        # get the unique labels
        unique_labels = np.unique(labels)
        # get the number of samples for each label
        label_counts = [np.sum(labels == label) for label in unique_labels]
        # calculate the number of samples for each label for each set
        train_label_counts = [np.sum(train_labels == label) for label in unique_labels]
        train_temp_label_counts = [np.sum(train_temp_labels == label) for label in unique_labels]
        val_label_counts = [np.sum(val_labels == label) for label in unique_labels]
        test_label_counts = [np.sum(test_labels == label) for label in unique_labels]
        # calculate the total number of samples for each setf
        total_count = len(labels)
        train_total_count = len(train_labels)
        train_temp_total_count = len(train_temp_labels)
        val_total_count = len(val_labels)
        test_total_count = len(test_labels)

        msg = f'debug stratified_split_dataset: \n' + \
                f'unique_labels: {unique_labels} \n' + \
                f'label_counts: {label_counts} \n' + \
                f'total_count: {total_count} \n' + \
                f'train_temp_label_counts: {train_temp_label_counts} \n' + \
                f'train_temp_total_count: {train_temp_total_count} \n' + \
                f'train_label_counts: {train_label_counts} \n' + \
                f'train_total_count: {train_total_count} \n' + \
                f'val_label_counts: {val_label_counts} \n' + \
                f'val_total_count: {val_total_count} \n' + \
                f'test_label_counts: {test_label_counts} \n' + \
                f'test_total_count: {test_total_count} \n'
        print(msg)

    # Creating subsets for training, validation, and test sets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Disable augmentation for validation and test datasets
    if isinstance(val_dataset, torch.utils.data.Subset):
        val_dataset.dataset.random_settings['augmentation'] = False
    if isinstance(test_dataset, torch.utils.data.Subset):
        test_dataset.dataset.random_settings['augmentation'] = False

    return train_dataset, val_dataset, test_dataset, positive_values

def balance_dataset_old(train_idx, train_labels, label_type, debug=False):
    """
    Balance the dataset by performing stratified repeated sampling.

    Args:
        train_idx (np.ndarray): Indices of the training samples.
        labels (np.ndarray): Array of labels corresponding to the samples.
        label_type (str): Type of label ('HRP5' or 'HRP2').
        debug (bool): If True, enables debugging output. Default is False.

    Returns:
        np.ndarray: Updated indices of the training samples.
        torch.Tensor: Positive values for weight calculation.
    """
    train_unique_labels = np.unique(train_labels)
    train_label_counts = [np.sum(train_labels == label) for label in train_unique_labels]
    train_total_count = np.sum(train_label_counts)

    if debug:
        msg = f'debug balance: \n' + \
              f'train_unique_labels: {train_unique_labels} \n' + \
              f'train_label_counts: {train_label_counts} \n' + \
              f'train_total_count: {train_total_count} \n'
        print(msg)

    if label_type == 'HRP5':
        non_hrp_count = train_label_counts[0]
        hrp_count = train_total_count - non_hrp_count
        diff = np.abs(non_hrp_count - hrp_count)
        inverse_sum = np.sum(1 / np.array(train_label_counts[1:]))

        if debug:
            msg = f'debug balance HRP5: \n' + \
                  f'non_hrp_count: {non_hrp_count} \n' + \
                  f'hrp_count: {hrp_count} \n' + \
                  f'diff: {diff} \n'
            print(msg)

        for i in range(1, len(train_unique_labels)):
            case_label = train_unique_labels[i]
            case_count = train_label_counts[i]
            repeat_times = int((diff * 1 / case_count) / inverse_sum)
            matching_indices = np.where(train_labels == case_label)[0]
            case_idx = train_idx[matching_indices]
            repeated_idx = np.random.choice(case_idx, repeat_times)
            train_idx = np.concatenate((train_idx, repeated_idx))

            if debug:
                repeated_labels_temp = [train_labels[i] for i in repeated_idx]
                repeated_label_counts_temp = [np.sum(repeated_labels_temp == label) for label in train_unique_labels]
                train_labels_temp = [train_labels[i] for i in train_idx]
                train_label_counts_temp = [np.sum(train_labels_temp == label) for label in train_unique_labels]
                msg = f'debug balance HRP5: \n' + \
                      f'case_label: {case_label} \n' + \
                      f'case_count: {case_count} \n' + \
                      f'repeat_times: {repeat_times} \n' + \
                      f'repeated_label_counts_temp: {repeated_label_counts_temp} \n' + \
                      f'train_label_counts_temp: {train_label_counts_temp} \n'
                print(msg)

        positive_values = torch.tensor([np.sum(train_labels[train_idx] == label) for label in train_unique_labels[1:]])

    elif label_type == 'HRP2':
        non_hrp_count = train_label_counts[0]
        hrp_count = train_total_count - non_hrp_count
        diff = np.abs(non_hrp_count - hrp_count)

        if debug:
            msg = f'debug balance HRP2: \n' + \
                  f'non_hrp_count: {non_hrp_count} \n' + \
                  f'hrp_count: {hrp_count} \n' + \
                  f'diff: {diff} \n'
            print(msg)

        case_label = train_unique_labels[1]
        case_count = train_label_counts[1]
        repeat_times = diff
        matching_indices = np.where(train_labels == case_label)[0]
        case_idx = train_idx[matching_indices]
        repeated_idx = np.random.choice(case_idx, repeat_times)
        train_idx = np.concatenate((train_idx, repeated_idx))

        positive_values = torch.tensor([np.sum(train_labels[train_idx] == label) for label in train_unique_labels[1:]])

        if debug:
            repeated_labels_temp = [train_labels[np.where(train_idx==i)[0][0]] for i in repeated_idx]
            repeated_label_counts_temp = [np.sum(repeated_labels_temp == label) for label in train_unique_labels]
            train_labels_temp = [train_labels[np.where(train_idx==i)[0][0]] for i in train_idx]
            train_label_counts_temp = [np.sum(train_labels_temp == label) for label in train_unique_labels]
            msg = f'debug balance HRP2: \n' + \
                  f'case_label: {case_label} \n' + \
                  f'case_count: {case_count} \n' + \
                  f'repeat_times: {repeat_times} \n' + \
                  f'repeated_label_counts_temp: {repeated_label_counts_temp} \n' + \
                  f'train_label_counts_temp: {train_label_counts_temp} \n' + \
                  f'positive_values: {positive_values} \n'
            print(msg)

    return train_idx, positive_values

def balance_dataset(train_idx, train_labels, label_type, debug=False):
    """
    Balance the dataset by performing stratified repeated sampling.

    Args:
        train_idx (np.ndarray): Indices of the training samples.
        train_labels (np.ndarray): Array of labels corresponding to the samples.
        label_type (str): Type of label ('HRP5' or 'HRP2').
        debug (bool): If True, enables debugging output. Default is False.

    Returns:
        np.ndarray: Updated indices of the training samples.
        torch.Tensor: Positive values for weight calculation.
    """
    # Get unique labels and their counts
    train_unique_labels = np.unique(train_labels)
    train_label_counts = [np.sum(train_labels == label) for label in train_unique_labels]
    train_total_count = np.sum(train_label_counts)

    if debug:
        print(f"\n=== balance_dataset debugging ... ===")
        print(f"*Initial Dataset Statistics*")
        print(f"Unique Labels: {train_unique_labels}")
        print(f"Label Counts: {train_label_counts}")
        print(f"Total Samples: {train_total_count}\n")

    # Separate HRP5 and HRP2 balancing logic
    if label_type == 'HRP5':
        non_hrp_count = train_label_counts[0]
        hrp_count = train_total_count - non_hrp_count
        diff = non_hrp_count - hrp_count

        # inverse repetition
        inverse_sum = np.sum(1 / np.array(train_label_counts[1:]))

        if debug:
            print(f"*Balancing HRP5 Labels*")
            print(f"Non-HRP Count: {non_hrp_count}")
            print(f"HRP Count: {hrp_count}")
            print(f"Difference (Imbalance): {diff}\n")

        # Loop through each HRP class and repeat indices as needed
        for i in range(1, len(train_unique_labels)):
            case_label = train_unique_labels[i]
            case_count = train_label_counts[i]

            if diff > 0:  # Only perform balancing if there is an imbalance
                # repeat times
                repeat_times_inverse = int((diff * 1 / case_count) / inverse_sum)
                repeat_times = min(repeat_times_inverse, case_count * 3)  # Limit the number of repeats to prevent over-repetition

                matching_indices = np.where(train_labels == case_label)[0]
                case_idx = train_idx[matching_indices]

                # Ensure safe sampling
                if len(case_idx) > 0:
                    repeated_idx = np.random.choice(case_idx, repeat_times, replace=True)
                    train_idx = np.concatenate((train_idx, repeated_idx))

                    if debug:
                        print(f"Label: {case_label}")
                        print(f"Original Count: {case_count}")
                        print(f"Repeat Times: {repeat_times}")
                        print(f"New Training Indices Length: {len(train_idx)}\n")
            
        # Calculate positive values for weight calculation (HRP classes only)
        new_train_labels = [train_labels[i] for i in train_idx]
        positive_values = torch.tensor([np.sum(new_train_labels == label) for label in train_unique_labels[1:]])

    elif label_type == 'HRP2':
        non_hrp_count = train_label_counts[0]
        hrp_count = train_total_count - non_hrp_count
        diff = non_hrp_count - hrp_count

        if debug:
            print(f"*Balancing HRP2 Labels*")
            print(f"Non-HRP Count: {non_hrp_count}")
            print(f"HRP Count: {hrp_count}")
            print(f"Difference (Imbalance): {diff}\n")

        # Balancing HRP2 classes - either increase HRP or non-HRP counts
        if diff > 0:
            # Case where non-HRP is greater - increase HRP counts
            case_label = train_unique_labels[1]
            case_count = train_label_counts[1]
            repeat_times = diff
            matching_indices = np.where(train_labels == case_label)[0]
            case_idx = train_idx[matching_indices]

            # Ensure safe sampling
            if len(case_idx) > 0:
                repeated_idx = np.random.choice(case_idx, repeat_times, replace=True)
                train_idx = np.concatenate((train_idx, repeated_idx))

                if debug:
                    print(f"Label: {case_label}")
                    print(f"Original Count: {case_count}")
                    print(f"Repeat Times: {repeat_times}")
                    print(f"New Training Indices Length: {len(train_idx)}")
                    print(f"New Training Indices: {train_idx}\n")

        elif diff < 0:
            # Case where HRP count is greater - increase non-HRP counts
            diff = abs(diff)
            case_label = train_unique_labels[0]  # non-HRP label
            case_count = train_label_counts[0]
            repeat_times = diff
            matching_indices = np.where(train_labels == case_label)[0]
            case_idx = train_idx[matching_indices]

            # Ensure safe sampling
            if len(case_idx) > 0:
                repeated_idx = np.random.choice(case_idx, repeat_times, replace=True)
                train_idx = np.concatenate((train_idx, repeated_idx))

                if debug:
                    print(f"Label: {case_label} (Non-HRP)")
                    print(f"Original Count: {case_count}")
                    print(f"Repeat Times: {repeat_times}")
                    print(f"New Training Indices Length: {len(train_idx)}")
                    print(f"New Training Indices: {train_idx}\n")

        # Calculate positive values for weight calculation (HRP classes only)
        new_train_labels = [train_labels[i] for i in train_idx]
        positive_values = torch.tensor([np.sum(new_train_labels == label) for label in train_unique_labels])

    else:
        raise ValueError("Unsupported label type. Use 'HRP5' or 'HRP2'.")

    if debug:
        # Final dataset statistics after balancing
        print(f"*Final Balanced Dataset Statistics*")
        print(f"Final Label Counts: {positive_values}")
        print(f"Total Samples After Balancing: {len(train_idx)}\n")

    return train_idx, positive_values

def get_files(directory, end=".nii.gz"):
    files_list = []
    directories_list = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(end):
                # Construct full file path
                full_path = os.path.join(root, file)
                files_list.append(full_path)
                
                # Get directory name once from root
                directory_name = os.path.basename(root)
                directories_list.append(directory_name)

    return directories_list, files_list

def get_label_calculation(df, filepath, driectory, label='HRP5', result_list=['posremod', 'lowatten', 'napkin', 'spotty', 'punctate'], debug=False):
    # get 'ID'
    ID_value = driectory
    # Convert the 'ID' column from integers to strings
    df['ID'] = df['ID'].astype(str)
    
    # get artery type
    file_name = os.path.basename(filepath)
    artery = ''
    for i in ['RCA', 'LAD', 'LCX']:
        if i in file_name: artery = i
        else: continue
    
    # get label
    if label=='Ca5':
        # output list
        out_label = [0] * len(result_list)
        result = df.loc[df['ID'] == ID_value, artery+'_'+'ca']
        out_label[int(result.iloc[0])] = 1
        out_label[int(len(result_list)-1)] = df.loc[df['ID'] == ID_value, artery+'_'+'stent'] # add stent label
        
        msg = f'\nlabel: {label}; ' + \
        f'file_name: {file_name}; ' + \
        f'Ca: {result.iloc[0]}; ' + \
        f'out_label: {out_label} \n'
    
    elif label=='Ca3':
        # output list
        out_label = [0] * 3
        result = df.loc[df['ID'] == ID_value, artery+'_'+'ca']
        if int(result.iloc[0]) >= 2:
            out_label[1] = 1
        else: out_label[0] = 1
        out_label[2] = df.loc[df['ID'] == ID_value, artery+'_'+'stent'] # add stent label

        msg = f'\nlabel: {label}; ' + \
        f'file_name: {file_name}; ' + \
        f'Ca: {result.iloc[0]}; ' + \
        f'out_label: {out_label} \n'

    elif label=='HRP5':
        out_label = [0] * len(result_list)
        for i in range(len(result_list)):
            plaque = str(result_list[i])
            result = df.loc[df['ID'] == ID_value, artery+'_'+plaque]
            out_label[i] = int(result)
        
        msg = f'\nlabel: {label}; ' + \
        f'file_name: {file_name}; ' + \
        f'result_list: {result_list}; ' + \
        f'ori label: {[df.loc[df["ID"] == ID_value, artery+"_"+plaque] for plaque in result_list]}; ' + \
        f'out_label: {out_label} \n'
    
    elif label=='HRP2':
        out_label = [0] * 2
        if np.sum([df.loc[df["ID"] == ID_value, artery+'_'+plaque] for plaque in result_list]) > 0:
            out_label[1] = 1
        else: out_label[0] = 1

        msg = f'\nlabel: {label}; ' + \
        f'file_name: {file_name}; ' + \
        f'result_list: {result_list}; ' + \
        f'ori label: {[df.loc[df["ID"] == ID_value, artery+"_"+plaque] for plaque in result_list]}; ' + \
        f'out_label: {out_label} \n'

    else:
        raise ValueError(f"Input label {label} is not in ['Ca3', 'Ca5', 'HRP5', 'HRP2']")
    
    out_label = torch.tensor(out_label, dtype=torch.float)
    
    # message
    if debug:
        print(msg)
    
    return out_label

def get_label_from_df(df, filepath, label='HRP5', debug=False):
    # file name
    file_name_tmp = os.path.basename(filepath)

    # Handle different file types
    if 'nii.gz' in file_name_tmp:
        file_name = file_name_tmp
    elif '.pt' in file_name_tmp:
        # 1) Extract base name before the first '.' => e.g. "110002_LADmsk25"
        base_name = file_name_tmp.split('.')[0]

        # 2) Use a regex to remove unwanted suffixes (msk, msk + digits, rot, rot + digits, etc.).
        #    This pattern looks for either "msk" or "rot", optionally followed by digits,
        #    at the END of the string ($).
        #    e.g. "110002_LADmsk25" => "110002_LAD"
        pattern = r'(msk\d*|rot\d*)$'
        base_name = re.sub(pattern, '', base_name)

        # 3) Reconstruct final filename with .nii.gz
        file_name = base_name + '.nii.gz'
    else:
        raise ValueError(f"Unsupported file type for file {file_name_tmp}")

    # 4) Retrieve label from dataframe
    result = df.loc[df['file'] == file_name, label]
    if result.empty:
        raise ValueError(f"Label {label} not found for file {file_name}")

    out_label = result.values[0]  # Extract the first element

    # 5) Convert from string -> actual list if needed
    if isinstance(out_label, str):
        out_label = ast.literal_eval(out_label)

    # 6) Convert to torch tensor
    out_label = torch.tensor(out_label, dtype=torch.float)

    if debug:
        print(f"\nfile_name: {file_name_tmp} -> final lookup: {file_name}; label: {out_label}")

    return out_label

def get_res_layer(filepath, layer_path="/home/jay/Documents/VIT_CTA/Vit_CT_Repository/Output_Folder/features_resnet", layer_end="_features.pt", debug=False):
    # file name
    file_name_tmp = os.path.basename(filepath)

    # Handle different file types
    if 'nii.gz' in file_name_tmp:
        file_name = file_name_tmp
        # replace .nii.gz with ''
        base_name = file_name.replace('.nii.gz', '')
    elif '.pt' in file_name_tmp:
        # 1) Extract base name before the first '.' => e.g. "110002_LADmsk25"
        base_name = file_name_tmp.split('.')[0]

        # 2) Use a regex to remove unwanted suffixes (msk, msk + digits, rot, rot + digits, etc.).
        #    This pattern looks for either "msk" or "rot", optionally followed by digits,
        #    at the END of the string ($).
        #    e.g. "110002_LADmsk25" => "110002_LAD"
        pattern = r'(msk\d*|rot\d*)$'
        base_name = re.sub(pattern, '', base_name)
    else:
        raise ValueError(f"Unsupported file type for file {file_name_tmp}")

    # 3) Reconstruct final filename with layer_end
    layer_file = base_name + layer_end
    file_name = os.path.join(layer_path, layer_file)

    # 4) check if the file exists
    if not os.path.exists(file_name):
        raise ValueError(f"File {file_name} does not exist")
    else:
        out_label = torch.load(file_name)

    if debug:
        print(f"\nfile_name: {file_name_tmp} -> final lookup: {file_name}; label: {out_label}")

    return out_label

def get_center_position(filepath, folder_path = '/media/jay/Storage21/ScotHeart_Aug_ResVit/positions', file_end="_pos.pt", debug=False):
    # file name
    file_name_tmp = os.path.basename(filepath)

    # Handle different file types
    if 'nii.gz' in file_name_tmp:
        file_name = file_name_tmp
        # replace .nii.gz with ''
        base_name = file_name.replace('.nii.gz', '')
    elif '.pt' in file_name_tmp:
        # 1) Extract base name before the first '.' => e.g. "110002_LADmsk25"
        base_name = file_name_tmp.split('.')[0]

        # 2) Use a regex to remove unwanted suffixes (msk, msk + digits, rot, rot + digits, etc.).
        #    This pattern looks for either "msk" or "rot", optionally followed by digits,
        #    at the END of the string ($).
        #    e.g. "110002_LADmsk25" => "110002_LAD"
        pattern = r'(msk\d*|rot\d*)$'
        base_name = re.sub(pattern, '', base_name)
    else:
        raise ValueError(f"Unsupported file type for file {file_name_tmp}")

    # 3) file name
    file_name = base_name + file_end
    file_path = os.path.join(folder_path, file_name)

    # 4) check if the file exists, load tensor
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    else:
        out_label = torch.load(file_path)

    if debug:
        print(f"\nfile_name: {file_name_tmp} -> final lookup: {file_name}; label: {out_label}")

    return out_label

def get_unet_info(filepath, folder_path = '/media/jay/Storage2/ScotHeart_Aug_ResVit/unet', file_end="_grid_features.pt", debug=False):
    # file name
    file_name_tmp = os.path.basename(filepath)

    # Handle different file types
    if 'nii.gz' in file_name_tmp:
        file_name = file_name_tmp
        # replace .nii.gz with ''
        base_name = file_name.replace('.nii.gz', '')
    elif '.pt' in file_name_tmp:
        # 1) Extract base name before the first '.' => e.g. "110002_LADmsk25"
        base_name = file_name_tmp.split('.')[0]

        # 2) Use a regex to remove unwanted suffixes (msk, msk + digits, rot, rot + digits, etc.).
        #    This pattern looks for either "msk" or "rot", optionally followed by digits,
        #    at the END of the string ($).
        #    e.g. "110002_LADmsk25" => "110002_LAD"
        pattern = r'(msk\d*|rot\d*)$'
        base_name = re.sub(pattern, '', base_name)
    else:
        raise ValueError(f"Unsupported file type for file {file_name_tmp}")

    # 3) file name
    file_name = base_name + file_end
    file_path = os.path.join(folder_path, file_name)

    # 4) check if the file exists, load tensor
    if not os.path.exists(file_path):
        if debug: raise ValueError(f"File {file_path} does not exist")
        else: out_label = torch.ones([196, 256])
    else:
        out_label = torch.load(file_path)

    if debug:
        print(f"\nfile_name: {file_name_tmp} -> final lookup: {file_name}; label: {out_label}")

    return out_label

def slices_thresholding(filenames, directories, df, slices_thres_dict, debug=False):
    """
    Filters filenames and directories based on slice thresholds.

    Parameters:
    - filenames: list of file paths to be filtered
    - directories: list of directory IDs corresponding to filenames
    - df: DataFrame containing information about files and slices
    - slices_thres_dict: Dictionary specifying the threshold for each artery
    - debug: Boolean flag to print debug information

    Returns:
    - filtered_filenames: list of filtered file paths
    - filtered_directories: list of filtered directory IDs
    """
    # Extract base filenames from the provided file paths
    base_filenames = [os.path.basename(filepath) for filepath in filenames]

    # Convert the 'ID' column from integers to strings
    df['ID'] = df['ID'].astype(str)
    
    # List to hold files that need to be removed
    remove_files = []
    
    # Iterate over each artery and directory to apply thresholds
    for artery in list(slices_thres_dict.keys()):
        for ID_value in directories:
            slice_value = int(df.loc[df['ID'] == ID_value, artery + ' Slices'])
            if slice_value <= slices_thres_dict[artery]:
                remove_files.append(ID_value + '_' + artery + '.nii.gz')

    # Find indices of elements in `remove_files` that are in `base_filenames`
    indices_to_remove = [base_filenames.index(value) for value in remove_files if value in base_filenames]
    
    # Remove elements at the found indices from `filenames` and `directories`
    filtered_filenames = [filenames[i] for i in range(len(filenames)) if i not in indices_to_remove]
    filtered_directories = [directories[i] for i in range(len(directories)) if i not in indices_to_remove]
    
    # Print debug information if enabled
    if debug:
        msg = f'slices_thres_dict: {slices_thres_dict}\n' + \
              f'remove_files: {remove_files} \n' + \
              f'indices_to_remove: {indices_to_remove}\n' + \
              f'filtered_filenames: {filtered_filenames}\n' + \
              f'filtered_directories: {filtered_directories}\n'
        print(msg)
    
    return filtered_filenames, filtered_directories

def stratification_weight_generate(filenames, get_label_func, label, debug=False, weight_type='bce'):
    # tensor size
    tensor_size = get_label_func(filenames[0]).shape[0]

    # Initialize variables
    total_files = torch.tensor([int(len(filenames))] * tensor_size)
    positive_values = torch.tensor([0] * tensor_size)
    all_values = np.zeros((len(filenames), tensor_size))
    labels = np.zeros(len(filenames))
    labels_mask = np.ones(len(filenames))

    # Counting positive values
    for i in range(len(filenames)):
        output_label = get_label_func(filenames[i])
        positive_values = torch.add(output_label, positive_values)
        all_values[i] = output_label.cpu().numpy()

    # Stratification of labels
    all_values_sum = all_values.sum(axis=0)
    sorted_indices = np.argsort(all_values_sum)
    for index in sorted_indices:
        if label == 'HRP5': # label = 0 is healthy case [0, 1, 2, ...]
            labels += all_values[:, index] * labels_mask * (index + 1)
        else:
            labels += all_values[:, index] * labels_mask * index
        labels_mask = 1 * ((labels_mask - all_values[:, index]) == 1)

    if debug:
        print(f"=====Stratification weight debug:=====\n"
              f"all_values_sum: {all_values_sum}\n"
              f"total_files: {total_files}\n"
              f"sorted_indices: {sorted_indices}\n"
              f"labels first 20: {labels[:20]}")

    return weight_generate(positive_values, debug=debug), labels

def weight_generate(positive_values, debug=False):
    # Replace inf with 1
    weights = torch.divide(max(positive_values), positive_values)
    weights = torch.where(torch.isinf(weights), torch.tensor(1.0), weights)

    if debug:
        print(f"=====Weight generation debug:=====\n"
            f"positive_values: {positive_values}\n"
            f"weights: {weights}")

    return weights

def stratified_k_fold_split_and_balance(internal_idx, internal_labels, fold=5, balance=True, in_weights=torch.tensor, label='HRP2', debug=False):
    """
    Performs stratified K-Fold split and balances the training dataset for each fold.

    Parameters:
    - dataset: Dataset object to split.
    - internal_idx: List of indices for the internal dataset.
    - internal_labels: List of labels for the internal dataset.
    - fold: Number of folds for Stratified K-Fold cross-validation.
    - balance: Boolean flag to balance the training data in each fold.
    - label: The target label used for training.
    - debug: Boolean flag for enabling debug mode.

    Returns:
    - fold_idx_dict: Dictionary containing training and validation indices for each fold.
    - weights: List of weights for each fold.
    """
    from sklearn.model_selection import StratifiedKFold

    fold_idx_dict = {}
    weights = [in_weights] * fold

    if debug:
        print(f"=====StratifiedKFold debug:=====\n")
        print(f'weights initial: {weights}, length: {len(weights)}')

    import utils.config
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=utils.config.RANDOM_STATE)
    for i, (train_idx, val_idx) in enumerate(skf.split(internal_idx, internal_labels)):
        fold_idx_dict[f'train_{i}'] = np.array([internal_idx[idx] for idx in train_idx])
        fold_idx_dict[f'val_{i}'] = np.array([internal_idx[idx] for idx in val_idx])

        if debug:
            print(f"Fold {i}: Length of train indices: {len(fold_idx_dict[f'train_{i}'])}, Length of validation indices: {len(fold_idx_dict[f'val_{i}'])}")
            # print(f"Fold {i}:\n"
            #       f"Training indices: {fold_idx_dict[f'train_{i}']},\n"
            #       f"Validation indices: {fold_idx_dict[f'val_{i}']}")
            # print(f"Fold {i}:\n"
            #       f'Train_idx: {train_idx}\n'
            #       f'Val_idx: {val_idx}\n')

        # Balance the dataset for each fold's training data
        if balance:
            internal_labels_use = [internal_labels[idx] for idx in train_idx]
            internal_idx_use = np.arange(len(train_idx))
            internal_idx_balance, positive_values = balance_dataset(internal_idx_use, internal_labels_use, label, debug=debug)
            fold_training_idx = [train_idx[idx] for idx in internal_idx_balance]

            weight = weight_generate(positive_values, debug=debug)  # Generate weights for this fold
            weights[i] = weight

            if debug:
                print(f"Fold {i}: Balanced training indices length: {len(fold_training_idx)}")
                print(f"Fold {i}: Weights generated: {weight}")
        else:
            fold_training_idx = train_idx

        fold_idx_dict[f'train_{i}'] = np.array([internal_idx[idx] for idx in fold_training_idx])

        if debug:
            new_train_labels = [internal_labels[idx] for idx in train_idx]
            val_labels = [internal_labels[idx] for idx in val_idx]
            print(f"Fold {i}: Final training indices length: {len(fold_idx_dict[f'train_{i}'])}")
            # print(f'Training indices: {fold_idx_dict[f"train_{i}"]}')
            print(f'Training lables positive: {np.sum(new_train_labels)}')
            print(f'Training lables negative: {len(new_train_labels) - np.sum(new_train_labels)}')
            # print(f'Training labels: {new_train_labels}')
            print(f"Fold {i}: Final validation indices length: {len(fold_idx_dict[f'val_{i}'])}")
            # print(f'Validation indices: {fold_idx_dict[f"val_{i}"]}')
            print(f'Validation labels positive: {np.sum(val_labels)}')
            print(f'Validation labels negative: {len(val_labels) - np.sum(val_labels)}')
            # print(f'Validation labels: {val_labels}\n')

    return fold_idx_dict, weights


if __name__ == "__main__":
    # Mock data
    train_idx = np.array(range(100, 120))
    train_labels = np.array([0] * 4 + [1] * 16)  # Imbalanced dataset

    # Balance dataset
    # train_idx, positive_values = balance_dataset(train_idx, train_labels, 'HRP2', debug=True)
    # print(f"Balanced training indices: {train_idx}")
    # print(f'Type train_idx: {type(train_idx)}')
    # print(f"Positive values: {positive_values}")

    # Stratified split dataset
    fold_idx_dict, weights = stratified_k_fold_split_and_balance(
        train_idx, 
        train_labels, 
        fold=5, 
        balance=True, 
        label='HRP2', 
        debug=True)
    print(f'weights: {weights}')                                                    
