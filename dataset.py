import numpy as np
import torch
# def LoadData():
    
#     input_folder = "/home/maggie/test/Pangu-Weather-pytorch/input_data"
#     print(input_folder)
#     input_surface_path = f"{input_folder}/input_surface.npy"
#     input_upper_path = f"{input_folder}/input_upper.npy"

#     surface_data = np.load(input_surface_path).astype(np.float32)
#     upper_air_data = np.load(input_upper_path).astype(np.float32)

#     return surface_data, upper_air_data


# def LoadConstantMask():
#     """
#     self.land_mask, self.soil_type, self.topography = LoadConstantMask()
#     """
#     base_path = "./constant_masks"
    
#     # Construct the full file paths
#     land_mask_file = f"{base_path}/land_mask.npy"
#     soil_type_file = f"{base_path}/soil_type.npy"
#     topography_file = f"{base_path}/topography.npy"

#     # Load the masks
#     land_mask = np.load(land_mask_file)
#     soil_type = np.load(soil_type_file)
#     topography = np.load(topography_file)

#     return land_mask, soil_type, topography

# def LoadData(step):
#     # dummy load for each step, we have upper air and surface input
#     inputs = torch.randn(13, 1440, 721, 5)
#     inputs_surface = torch.randn(1440, 721, 4)

#     targets = torch.randn(4, 13, 1440, 721, 5)
#     targets_surface = torch.randn(4, 1440, 721, 4)


#     return inputs, inputs_surface, targets, targets_surface

# def LoadConstantMask():
#     return torch.rand((1440, 721)), torch.rand((1440, 721)), torch.rand((1440, 721))

def LoadData(step):
    # dummy load for each step, we have upper air and surface input, no batch here, will unsequeeze when batch dimension is needed later
    inputs = torch.ones((13, 1440, 721, 5))
    inputs_surface = torch.ones((1440, 721, 4))

    targets = torch.ones((4, 13, 1440, 721, 5))
    targets_surface = torch.ones((4, 1440, 721, 4))

    return inputs, inputs_surface, targets, targets_surface

def LoadConstantMask():
    return torch.ones((1440, 721)), torch.ones((1440, 721)), torch.ones((1440, 721))