import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)


def render(obj_path, resolution=512, n_views=36, device=torch.device("cpu")):
    mesh = load_objs_as_meshes([obj_path], device=device)
    views = [[0, int(i * 360 / n_views)] for i in range(n_views)]

    images = []
    for i in range(len(views)):
        R, T = look_at_view_transform(1.5, views[i][0], views[i][1])
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = AmbientLights(device=device)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
            )
        )
        image = renderer(mesh)
        image = Image.fromarray((image[..., :3].cpu().squeeze(0) * 255).numpy().astype(np.uint8))
        images.append(image)
    return images
