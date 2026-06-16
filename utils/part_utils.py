import torch
import numpy as np
import neural_renderer as nr
from main import path_config
from models import SMPL


class PartRenderer():
    """Renderer used to render segmentation masks and part segmentations.
    Internally it uses the Neural 3D Mesh Renderer for rendering masks and part segmentations.
    """

    def __init__(self, focal_length=5000., render_res=224):
        """
        Initializes the renderer with specified parameters.

        Args:
        - focal_length (float): The focal length of the camera.
        - render_res (int): The resolution of the rendered image.

        Initializes the Neural 3D Mesh Renderer, loads required SMPL model data,
        and initializes textures and cube parts for rendering.
        """
        # Parameters for rendering
        self.focal_length = focal_length
        self.render_res = render_res

        # Initialize Neural 3D mesh renderer
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=render_res,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=False)

        # Load SMPL model faces
        self.faces = torch.from_numpy(SMPL(path_config.SMPL_MODEL_DIR).faces.astype(np.int32)).cuda()

        # Load vertex textures
        textures = np.load(path_config.VERTEX_TEXTURE_FILE)
        self.textures = torch.from_numpy(textures).cuda().float()

        # Load cube parts for rendering
        self.cube_parts = torch.cuda.FloatTensor(np.load(path_config.CUBE_PARTS_FILE))

    def get_parts(self, parts, mask):
        """
        Process renderer part image to get body part indices.

        Args:
        - parts (torch.Tensor): Part segmentation tensor from the renderer.
        - mask (torch.Tensor): Mask indicating valid parts in the segmentation.

        Returns:
        - parts (torch.Tensor): Processed body part indices.
        """
        bn, c, h, w = parts.shape
        mask = mask.view(-1, 1)
        parts_index = torch.floor(100 * parts.permute(0, 2, 3, 1).contiguous().view(-1, 3)).long()
        parts = self.cube_parts[parts_index[:, 0], parts_index[:, 1], parts_index[:, 2], None]
        parts *= mask
        parts = parts.view(bn, h, w).long()
        return parts

    def __call__(self, vertices, camera):
        """
        Wrapper function for rendering process.

        Args:
        - vertices (torch.Tensor): 3D vertices of the mesh.
        - camera (torch.Tensor): Camera parameters.

        Returns:
        - mask (torch.Tensor): Rendered segmentation mask.
        - parts (torch.Tensor): Rendered part segmentation.
        """
        # Estimate camera parameters given a fixed focal length
        cam_t = torch.stack([camera[:, 1], camera[:, 2], 2 * self.focal_length / (self.render_res * camera[:, 0] + 1e-9)], dim=-1)
        batch_size = vertices.shape[0]

        # Initialize intrinsic camera matrix K
        K = torch.eye(3, device=vertices.device)
        K[0, 0] = self.focal_length
        K[1, 1] = self.focal_length
        K[0, 2] = self.render_res / 2.
        K[1, 2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)

        # Initialize rotation matrix R and faces
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)
        faces = self.faces[None, :, :].expand(batch_size, -1, -1)

        # Render using Neural 3D Mesh Renderer
        parts, _, mask = self.neural_renderer(vertices, faces,
                                              textures=self.textures.expand(batch_size, -1, -1, -1, -1, -1),
                                              K=K, R=R, t=cam_t.unsqueeze(1))

        # Process parts to get body part indices
        parts = self.get_parts(parts, mask)

        return mask, parts
