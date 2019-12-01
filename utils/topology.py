import torch
import math
import numpy as np
from scipy.spatial import Delaunay


def get_circle(batch_size, masks_size, num_points, device):
    half_dim = masks_size / 2
    half_width = half_dim
    half_height = half_dim

    vert = np.array([[
        half_width + math.floor(math.cos(2 * math.pi / num_points * x) * 10),
        half_height + math.floor(math.sin(2 * math.pi / num_points * x) * 10)]
        for x in range(0, num_points)])
    vert = (vert - half_dim) / half_dim

    tri = Delaunay(vert).simplices.copy()

    vert = torch.Tensor(vert)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face = torch.Tensor(tri)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)

    vert[:, :, :, 1] = -vert[:, :, :, 1]

    return vert, face
