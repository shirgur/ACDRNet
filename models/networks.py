import torch
from torch import nn
import torch.nn.functional as F

from backbones.unet import Encoder as unet_encoder, Decoder as unet_decoder
from backbones.resnet import resnet50 as resnet_encoder, Decoder as resnet_decoder
from utils.topology import get_circle
import neural_renderer as nr


__all__ = ['CircleNet']


class CircleNet(nn.Module):
    def __init__(self,
                 args):
        super(CircleNet, self).__init__()
        self.num_nudes = args.num_nodes
        self.dec_dim = args.dec_dim
        self.dec_size = args.dec_size
        self.image_size = args.image_size
        self.stages = args.stages

        if args.arch == 'resnet':
            kwargs = {'stages': self.stages}
            res_dims = [256, 512, 1024, 2048]
            self.backbone = resnet_encoder(pretrained=True, **kwargs)
            dec_skip_dims = [res_dims[i] for i in self.stages][::-1]
            self.disp = resnet_decoder(dec_skip_dims, 2, self.dec_dim, self.dec_size, drop=args.drop)
        elif args.arch == 'unet':
            self.backbone = unet_encoder(args.enc_dim, drop=args.drop)
            self.disp = unet_decoder(self.backbone.dims, drop=args.drop)

        self.texture_size = 2
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.renderer = nr.Renderer(camera_mode='look_at', image_size=self.image_size, light_intensity_ambient=1,
                                    light_intensity_directional=1, perspective=False)

    def forward(self, x, iter=3):
        features = self.backbone(x)

        nodes, faces = get_circle(x.shape[0], self.dec_size, self.num_nudes, x.device)

        output_masks = []
        output_points = []

        disp = torch.tanh(self.disp(features))

        for i in range(iter):
            if disp.shape[2:] != x.shape[2:]:
                disp = F.interpolate(disp, size=x.shape[2:], mode='bilinear')

            nodes[..., 1] = nodes[..., 1] * -1
            # Sample and move
            Pxx = F.grid_sample(disp[:, 0:1], nodes).transpose(3, 2)
            Pyy = F.grid_sample(disp[:, 1:2], nodes).transpose(3, 2)
            dP = torch.cat((Pxx, Pyy), -1)
            nodes = nodes + dP
            nodes[..., 1] = nodes[..., 1] * -1

            # Render mask
            z = torch.ones((nodes.shape[0], 1, nodes.shape[2], 1)).to(nodes.device)
            P3d = torch.cat((nodes, z), 3)
            P3d = torch.squeeze(P3d, dim=1)
            faces = torch.squeeze(faces, dim=1).to(nodes.device)
            mask = self.renderer(P3d, faces, mode='silhouettes').unsqueeze(1)

            # Stack outputs
            output_masks.append(mask)
            output_points.append(nodes)

        if self.training:
            return output_masks, output_points
        else:
            return output_masks[-1]
