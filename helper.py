import torch
import torch.nn as nn

import torch.nn.functional as F

def Pad2D(data, patch_size=(4, 4)):
    """
    Pad 2D surface input
    """
    _, latitude, _  = data.shape

    latitude_padding = (4 - latitude % 4) % 4
    padded_data = F.pad(data, (0, 0, 0, latitude_padding), mode='constant', value=0)


    return padded_data




def Pad3D(data, patch=(2,4,4)):
    """
    Pad upper air var.
    """
    try: # padding for patch emb
      height, longitude, latitude, _ = data.shape # pad height(13) and lat(721)
      # patch = (2,4,4)

    except: # padding for earth spec block
      _, height, longitude, latitude, _ = data.shape
      # patch = (2, 6, 12) # window patch

    height_patch = patch[0]
    longitude_patch = patch[1]
    latitude_patch = patch[2] # 181 pad 12


    height_padding = (height_patch - height % height_patch) % height_patch
    latitude_padding = (latitude_patch - latitude % latitude_patch) % latitude_patch
    print("latitude_padding", latitude_padding)
    # Apply padding
    padded_data = F.pad(data, (0, 0, 0, latitude_padding, 0, 0, 0, height_padding), mode='constant', value=0)

    return padded_data



def gen_mask(x):

    return

def no_mask():
  
    return