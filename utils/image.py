from typing import Tuple
import numpy as np
from scipy.ndimage import rotate, gaussian_filter, shift


def flip_height_width_diagonals(data_array: np.ndarray, 
                                setting={'random_flip': True, 'flip_height': True, 'flip_width': True, 'flip_diagonal': True, 'flip_antidiagonal': True}):
    """
    Flips the height (D1) and width (D2) of a 3D numpy array independently and randomly,
    including diagonal flips, while keeping the depth direction (D3) unchanged.

    Parameters:
    data_array (np.ndarray): The 3D numpy array to be flipped.
    setting (dict): A dictionary containing the flip settings.

    Returns:
    np.ndarray: The 3D numpy array with flipped height and/or width.
    """
    # Randomly decide whether to flip height
    if setting['random_flip']:
        if np.random.rand() > 0.5:
            data_array = np.flip(data_array, axis=0)  # Flip along the height (D1)

        # Randomly decide whether to flip width
        if np.random.rand() > 0.5:
            data_array = np.flip(data_array, axis=1)  # Flip along the width (D2)

        # Randomly decide whether to flip along the main diagonal
        if np.random.rand() > 0.5:
            data_array = np.transpose(data_array, axes=(1, 0, 2))  # Flip along the main diagonal of D1 and D2

        # Randomly decide whether to flip along the anti-diagonal
        if np.random.rand() > 0.5:
            data_array = np.flip(data_array, axis=0)  # Flip along the height (D1)
            data_array = np.flip(data_array, axis=1)  # Flip along the width (D2)
            data_array = np.transpose(data_array, axes=(1, 0, 2))  # Flip along the main diagonal of D1 and D2
    elif setting['flip_height']:
        data_array = np.flip(data_array, axis=0)
    elif setting['flip_width']:
        data_array = np.flip(data_array, axis=1)
    elif setting['flip_diagonal']:
        data_array = np.transpose(data_array, axes=(1, 0, 2))
    elif setting['flip_antidiagonal']:
        data_array = np.flip(data_array, axis=0)
        data_array = np.flip(data_array, axis=1)
        data_array = np.transpose(data_array, axes=(1, 0, 2))

    return data_array

def rotate_along_depth_random_angle(data_array: np.ndarray, angle_range: float):
    """
    Rotates a 3D numpy array along the depth direction (D3) by a random angle within a specified range.

    Parameters:
    data_array (np.ndarray): The 3D numpy array to be rotated.
    angle_range (tuple): A tuple specifying the range (min_angle, max_angle) for the random rotation angle.

    Returns:
    np.ndarray: The rotated 3D numpy array.
    """
    # Generate a random angle within the specified range
    angle = np.random.uniform(-angle_range, angle_range)

    # Rotate along the depth axis (D3)
    rotated_data_array = rotate(data_array, angle, axes=(1, 0), reshape=False)

    return rotated_data_array

def add_integer_white_noise_to_slices_range(data_array, noise_range=5):
    """
    Adds integer white noise within a specified range to each slice along the 3rd axis of the data array.

    Parameters:
    - nifti_img: The NIfTI image object from which to get the data array.
    - min_noise: The minimum value of the noise to be added.
    - max_noise: The maximum value of the noise to be added.

    Returns:
    - The data array with added integer white noise within the specified range.
    """
    # Iterate over each slice along the 3rd axis (assuming the 3rd axis is the depth)
    for i in range(data_array.shape[2]):
        # Generate integer white noise within the specified range with the same shape as the slice
        noise = np.random.randint(-abs(noise_range), abs(noise_range), size=data_array[:, :, i].shape)
        # Add the noise to the slice
        data_array[:, :, i] = data_array[:, :, i] + noise

    return data_array


def apply_gaussian_filter_to_slices(data_array, random=False, sigma=1.5):
    """
    Applies a Gaussian filter with a controllable scale to each slice along the 3rd axis of the data array.

    Parameters:
    - data_array: The data array extracted from a NIfTI image.
    - sigma: The standard deviation for Gaussian kernel. Default is 1.0.

    Returns:
    - The data array with the Gaussian filter applied to each slice.
    """
    # Copy the data array to avoid modifying the original data
    filtered_data_array = np.copy(data_array)

    if random:
        sigma = np.random.uniform(0, abs(sigma))

    # Iterate over each slice along the 3rd axis (assuming the 3rd axis is the depth)
    for i in range(filtered_data_array.shape[2]):
        # Apply the Gaussian filter to the slice
        filtered_data_array[:, :, i] = gaussian_filter(filtered_data_array[:, :, i], sigma=sigma)

    return filtered_data_array


def shift_array_along_axis(data_array, random=True, shift_1st=4, shift_2nd=4):
    """
    Shifts the array relative to the 3rd axis with a specific move in the 1st and 2nd directions.

    Parameters:
    - data_array: The data array extracted from a NIfTI image.
    - shift_1st: The amount to shift in the 1st direction (rows).
    - shift_2nd: The amount to shift in the 2nd direction (columns).

    Returns:
    - The data array with the specified shifts applied.
    """
    # Initialize an empty array with the same shape as the input data_array
    shifted_data_array = np.empty_like(data_array)

    if random:
        shift_1st = np.random.randint(-abs(shift_1st), abs(shift_1st))
        shift_2nd = np.random.randint(-abs(shift_2nd), abs(shift_2nd))

    # Iterate over each slice along the 3rd axis (assuming the 3rd axis is the depth)
    for i in range(data_array.shape[2]):
        # Apply the shift to the slice
        shifted_data_array[:, :, i] = shift(data_array[:, :, i], shift=[shift_1st, shift_2nd], mode='nearest')

    return shifted_data_array

def process_slice(i, img, start_x, end_x, start_y, end_y, random_range):
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

    return slice_img

def process_torch_slice(transform, img, i):
    return transform(img[:, :, i])