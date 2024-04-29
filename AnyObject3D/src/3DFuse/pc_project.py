import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from my.utils import tqdm

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras   
from pytorch3d.renderer import (
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,
)

import torch.nn.functional as F

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


class PointsRenderer(nn.Module):
    """
    Modified version of Pytorch3D PointsRenderer
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        
        # import pdb; pdb.set_trace()
        
        depth_map = fragments[1][0,...,:1]
        
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, depth_map


def render_depth_from_cloud(points, angles, raster_settings, device,calibration_value=0):
    
    radius = 2.3
    
    horizontal = angles[0]+calibration_value
    elevation = angles[1]
    FoV = angles[2]


    camera = py3d_camera(radius, elevation, horizontal, FoV, device)
    
    point_loc = torch.tensor(points.coords).to(device)
    colors = torch.tensor(np.stack([points.channels["R"], points.channels["G"], points.channels["B"]], axis=-1)).to(device)

    matching_rotation = torch.tensor([[[1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, -1.0, 0.0]]]).to(device)
    
    rot_points = (matching_rotation @ point_loc[...,None]).squeeze()      
    
    point_cloud = Pointclouds(points=[rot_points], features=[colors])
    
    _, raw_depth_map = pointcloud_renderer(point_cloud, camera, raster_settings, device)    

    disparity = camera.focal_length[0,0] / (raw_depth_map + 1e-9)
    
    max_disp = torch.max(disparity) 
    min_disp = torch.min(disparity[disparity > 0])
    
    norm_disparity = (disparity - min_disp) / (max_disp - min_disp)
    
    mask = norm_disparity > 0
    norm_disparity = norm_disparity * mask
    
    depth_map = F.interpolate(norm_disparity.permute(2,0,1)[None,...],size=512,mode='bilinear')[0]
    depth_map = depth_map.repeat(3,1,1)
    
    return depth_map


def py3d_camera(radius, elevation, horizontal, FoV, device, img_size=800):
    
    fov_rad = torch.deg2rad(torch.tensor(FoV))
    focal = 1 / torch.tan(fov_rad / 2) * (2. / 2)
    
    focal_length = torch.tensor([[focal,focal]]).float()
    image_size = torch.tensor([[img_size,img_size]]).double()

    
    R, T = look_at_view_transform(dist=radius, elev=elevation, azim=horizontal, degrees=True)

    
    camera = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_length,
            image_size=image_size,
            device=device,
        )

    return camera

def pointcloud_renderer(point_cloud, camera, raster_settings, device):

    camera = camera.to(device)

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    ).to(device)
    
    image = renderer(point_cloud)
    
    return image
    
def point_e(device,exp_dir):
    print('creating base model...')
    base_name = 'base1B' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    img = Image.open(os.path.join(exp_dir,'initial_image','instance0.png'))

    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
        
    pc = sampler.output_to_point_clouds(samples)[0]
    
    return pc


def point_e_gradio(img,device):
    #print(img)
    import gc
    print('creating base model...')
    base_name = 'base300M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], "cpu")
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    gc.collect()
    torch.cuda.empty_cache()
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], "cpu")
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    gc.collect()
    torch.cuda.empty_cache()
    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, "cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', "cpu"))
    # gc.collect()
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # base_model.time_embed = base_model.time_embed.cuda()
    # base_model.ln_pre = base_model.ln_pre.cuda()
    # base_model.backbone = base_model.backbone.cuda()
    # base_model.ln_post = base_model.ln_post.cuda()
    # base_model.input_proj = base_model.input_proj.cuda()
    # base_model.output_proj = base_model.output_proj.cuda()
    # base_model.clip.model = base_model.clip.model.cuda()
    # base_model.clip_embed = base_model.clip_embed.cuda()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # upsampler_model.time_embed = upsampler_model.time_embed.cuda()
    # upsampler_model.ln_pre = upsampler_model.ln_pre.cuda()
    # upsampler_model.backbone = upsampler_model.backbone.cuda()
    # upsampler_model.ln_post = upsampler_model.ln_post.cuda()
    # upsampler_model.input_proj = upsampler_model.input_proj.cuda()
    # upsampler_model.output_proj = upsampler_model.output_proj.cuda()
    # upsampler_model.clip.model = upsampler_model.clip.model.cuda()
    # upsampler_model.clip_embed = upsampler_model.clip_embed.cuda()
    # upsampler_model.cond_point_proj = upsampler_model.cond_point_proj.cuda()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print("creating sampler")
    sampler = PointCloudSampler(
        device="cpu",
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
    print(len(samples))
    pc = sampler.output_to_point_clouds(samples)[0]
        # gc.collect()
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # base_model.time_embed = base_model.time_embed.cuda()
    # base_model.ln_pre = base_model.ln_pre.cuda()
    # base_model.backbone = base_model.backbone.cuda()
    # base_model.ln_post = base_model.ln_post.cuda()
    # base_model.input_proj = base_model.input_proj.cuda()
    # base_model.output_proj = base_model.output_proj.cuda()
    # base_model.clip.model = base_model.clip.model.cuda()
    # base_model.clip_embed = base_model.clip_embed.cuda()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # upsampler_model.time_embed = upsampler_model.time_embed.cuda()
    # upsampler_model.ln_pre = upsampler_model.ln_pre.cuda()
    # upsampler_model.backbone = upsampler_model.backbone.cuda()
    # upsampler_model.ln_post = upsampler_model.ln_post.cuda()
    # upsampler_model.input_proj = upsampler_model.input_proj.cuda()
    # upsampler_model.output_proj = upsampler_model.output_proj.cuda()
    # upsampler_model.clip.model = upsampler_model.clip.model.cuda()
    # upsampler_model.clip_embed = upsampler_model.clip_embed.cuda()
    # upsampler_model.cond_point_proj = upsampler_model.cond_point_proj.cuda()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return pc
