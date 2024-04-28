import numpy as np

def LoadData():
    """
    LoadData: Load the ERA5 data
    """
    input_folder = "test/pangu_torch/data"
    input_surface_path = f"{input_folder}/input_surface.npy"
    input_upper_path = f"{input_folder}/input_upper.npy"

    surface_data = np.load(input_surface_path).astype(np.float32)
    upper_air_data = np.load(input_upper_path).astype(np.float32)

    return surface_data, upper_air_data


def LoadConstantMask(mask_file):
    """
    Load constant masks from a file, e.g., soil type mask.
    
    Parameters:
        mask_file (str): The file path to the mask file.

    Returns:
        np.array: A numpy array representing the mask.
    """
    # Load mask from a numpy file
    mask = np.load(mask_file).astype(bool)
    return mask