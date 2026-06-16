import json
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import trimesh
import numpy as np
from skimage.transform import resize
from torchvision.utils import make_grid
import torch.nn.functional as F
from main import path_config, constants
from models.smpl import get_smpl_faces, get_model_faces, get_model_tpose
from utils.densepose_methods_utils import DensePoseMethods
from .geometry_utils import convert_to_full_img_cam
from utils.imutils import crop
try:
    import math
    import pyrender
    from pyrender.constants import RenderFlags
except ModuleNotFoundError:
    print('Failed to import *pyrender*. Please ignore the warning if there is no need to render results.')


from pytorch3d.structures.meshes import Meshes
# from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    AmbientLights,
    PointLights,
    RasterizationSettings,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    HardGouraudShader,
    HardFlatShader,
    TexturesVertex
)
import logging
logger = logging.getLogger(__name__)

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        """
        Constructor for WeakPerspectiveCamera class.

        Args:
            scale (list or tuple): Scale factors [scale_x, scale_y].
            translation (list or tuple): Translation offsets [tx, ty].
            znear (float, optional): Near clipping plane. Defaults to pyrender.camera.DEFAULT_Z_NEAR.
            zfar (float, optional): Far clipping plane. If None, uses the renderer's default.
            name (str, optional): Name of the camera.

        """
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        """
        Compute the weak perspective projection matrix.

        Args:
            width (int, optional): Width of the viewport.
            height (int, optional): Height of the viewport.

        Returns:
            numpy.ndarray: 4x4 projection matrix.

        """
        P = np.eye(4)  # Initialize 4x4 identity matrix
        P[0, 0] = self.scale[0]  # Scale factor along x-axis
        P[1, 1] = self.scale[1]  # Scale factor along y-axis
        P[0, 3] = self.translation[0] * self.scale[0]  # Translate x-coordinate
        P[1, 3] = -self.translation[1] * self.scale[1]  # Translate y-coordinate (note the negative sign)
        P[2, 2] = -1  # Weak perspective projection with -1 on the z component

        return P


class PyRenderer:
    def __init__(self, resolution=(224, 224), orig_img=False, wireframe=False, scale_ratio=1., vis_ratio=1.):
        """
        Initialize PyRenderer object.

        Args:
        - resolution: tuple, resolution of the renderer (width, height)
        - orig_img: bool, whether to include original image in rendering
        - wireframe: bool, whether to render in wireframe mode
        - scale_ratio: float, scaling ratio for rendering resolution
        - vis_ratio: float, visibility ratio for rendering

        """
        self.resolution = (resolution[0] * scale_ratio, resolution[1] * scale_ratio)

        # Define faces dictionary with model types and their respective faces data
        self.faces = {
            'smpl': get_model_faces('smpl'),  # Example: get_model_faces function to retrieve model faces
        }

        self.orig_img = orig_img
        self.wireframe = wireframe

        # Initialize offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        self.vis_ratio = vis_ratio

        # Set up the scene with background color and ambient light
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Define lights in the scene
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=1)

        # Define light poses and add lights to the scene
        yrot = np.radians(120)  # Angle of lights in radians

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        # Define and add spot lights to the scene
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                                    innerConeAngle=np.pi / 3, outerConeAngle=np.pi / 2)

        light_pose[:3, 3] = [1, 2, 2]
        self.scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [-1, 2, 2]
        self.scene.add(spot_l, pose=light_pose)

        # Define colors dictionary for different materials
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.55, 0.4, 0.9]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }

    def __call__(self, verts, faces=None, img=np.zeros((224, 224, 3)), cam=np.array([1, 0, 0]),
                 focal_length=[5000, 5000], camera_rotation=np.eye(3), crop_info=None,
                 angle=None, axis=None, mesh_filename=None, color_type=None, color=[1.0, 1.0, 0.9], iwp_mode=True,
                 crop_img=True, mesh_type='smpl', scale_ratio=1., rgba_mode=False):
        """
        Perform rendering based on provided parameters.

        Args:
        - verts: ndarray, vertices of the mesh
        - faces: ndarray, faces of the mesh (optional)
        - img: ndarray, input image to be included in rendering
        - cam: ndarray, camera parameters
        - focal_length: list, focal length parameters
        - camera_rotation: ndarray, rotation matrix of the camera
        - crop_info: dict, information for cropping
        - angle: float, angle of rotation
        - axis: ndarray, axis of rotation
        - mesh_filename: str, filename to export the mesh
        - color_type: str, type of color for rendering
        - color: list, color parameters
        - iwp_mode: bool, whether to use IWP mode
        - crop_img: bool, whether to crop the image
        - mesh_type: str, type of mesh to render
        - scale_ratio: float, scaling ratio for rendering resolution
        - rgba_mode: bool, whether to use RGBA mode

        Returns:
        - ndarray or list of ndarrays, rendered image(s)

        """
        if faces is None:
            faces = self.faces[mesh_type]

        # Create Trimesh object from vertices and faces
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Apply rotation transformation to mesh
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        # Export mesh if filename is provided
        if mesh_filename is not None:
            mesh.export(mesh_filename)

        # Apply rotation if angle and axis are provided
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        # Copy camera parameters for further modification
        cam = cam.copy()

        # Determine rendering resolution based on IWP mode
        if iwp_mode:
            resolution = np.array(img.shape[:2]) * scale_ratio
            if len(cam) == 4:
                sx, sy, tx, ty = cam
                camera_translation = np.array([- tx, ty, 2 * focal_length[0] / (resolution[0] * sy + 1e-9)])
            elif len(cam) == 3:
                sx, tx, ty = cam
                sy = sx
                camera_translation = np.array([- tx, ty, 2 * focal_length[0] / (resolution[0] * sy + 1e-9)])
            render_res = resolution
            self.renderer.viewport_width = render_res[1]
            self.renderer.viewport_height = render_res[0]
        else:
            if crop_info['opt_cam_t'] is None:
                camera_translation = convert_to_full_img_cam(
                    pare_cam=cam[None],
                    bbox_height=crop_info['bbox_scale'] * 200.,
                    bbox_center=crop_info['bbox_center'],
                    img_w=crop_info['img_w'],
                    img_h=crop_info['img_h'],
                    focal_length=focal_length[0],
                )
            else:
                camera_translation = crop_info['opt_cam_t']
            if torch.is_tensor(camera_translation):
                camera_translation = camera_translation[0].cpu().numpy()
            camera_translation = camera_translation.copy()
            camera_translation[0] *= -1
            if 'img_h' in crop_info and 'img_w' in crop_info:
                render_res = (int(crop_info['img_h'][0]), int(crop_info['img_w'][0]))
            else:
                render_res = img.shape[:2] if type(img) is not list else img[0].shape[:2]
            self.renderer.viewport_width = render_res[1]
            self.renderer.viewport_height = render_res[0]
            camera_rotation = camera_rotation.T

        # Define camera intrinsics based on focal length and rendering resolution
        camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                           cx=render_res[1] / 2., cy=render_res[0] / 2.)

        # Set material color based on color_type or provided color
        if color_type is not None:
            color = self.colors_dict[color_type]

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # Create pyrender Mesh object from Trimesh and apply material
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Add mesh to the scene and get mesh node
        mesh_node = self.scene.add(mesh, 'mesh')

        # Define camera pose based on rotation and translation
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation

        # Add camera to the scene and get camera node
        cam_node = self.scene.add(camera, pose=camera_pose)

        # Determine render flags based on wireframe mode
        if self.wireframe:
            render_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_WIREFRAME | pyrender.RenderFlags.SHADOWS_SPOT
        else:
            render_flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_SPOT

        # Perform rendering using the renderer and current scene configuration
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)

        # Crop rendered image if crop_info is provided and crop_img is True
        if crop_info is not None and crop_img:
            crop_res = img.shape[:2] if type(img) is not list else img[0].shape[:2]
            rgb, _, _ = crop(rgb, crop_info['bbox_center'][0], crop_info['bbox_scale'][0], crop_res)

        # Create a mask for valid pixels based on alpha channel
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]

        # Handle multiple input images in case img is a list
        image_list = [img] if type(img) is not list else img

        # Initialize list for storing returned images
        return_img = []

        # Iterate over each image in image_list
        for item in image_list:
            # Resize item if scale_ratio is not 1
            if scale_ratio != 1:
                orig_size = item.shape[:2]
                item = resize(item, (int(orig_size[0] * scale_ratio), int(orig_size[1] * scale_ratio)), anti_aliasing=True)
                item = (item * 255).astype(np.uint8)

            # Compute final output image using rendered rgb image and item
            output_img = rgb[:, :, :] * valid_mask * self.vis_ratio + (1 - valid_mask * self.vis_ratio) * item

            # Handle RGBA mode if enabled
            if rgba_mode:
                output_img_rgba = np.zeros((output_img.shape[0], output_img.shape[1], 4))
                output_img_rgba[:, :, :3] = output_img
                output_img_rgba[:, :, 3][valid_mask[:, :, 0]] = 255
                output_img = output_img_rgba.astype(np.uint8)

            # Convert output_img to uint8 format and append to return_img list
            image = output_img.astype(np.uint8)
            return_img.append(image)

            # Append original item to return_img list
            return_img.append(item)

        # If img is not a list, return_img should be the single processed image
        if type(img) is not list:
            return_img = return_img[0]

        # Remove mesh and camera nodes from the scene after rendering
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        # Return the processed image or list of images
        return return_img



class OpenDRenderer:
    def __init__(self, resolution=(224, 224), ratio=1):
        """
        Initialize the renderer with default parameters.

        Args:
            resolution (tuple): Initial resolution of the rendered image.
            ratio (float): Ratio to scale the resolution by.

        Attributes:
            resolution (tuple): Current resolution of the rendered image.
            ratio (float): Scaling ratio applied to the resolution.
            focal_length (float): Default focal length of the camera.
            K (np.ndarray): Camera intrinsic matrix.
            colors_dict (dict): Dictionary mapping color names to RGB arrays.
            renderer (ColoredRenderer): Renderer instance from ColoredRenderer class.
            faces (np.ndarray): Default face indices for rendering.
        """
        self.resolution = (resolution[0] * ratio, resolution[1] * ratio)
        self.ratio = ratio
        self.focal_length = 5000.
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                          [0., self.focal_length, self.resolution[0] / 2.],
                          [0., 0., 1.]])
        self.colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            'purple': np.array([0.5, 0.5, 0.7]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }
        self.renderer = ColoredRenderer()
        self.faces = get_smpl_faces()  # Assuming get_smpl_faces() returns face indices
    
    def reset_res(self, resolution):
        """
        Reset the renderer's resolution.

        Args:
            resolution (tuple): New resolution to set.

        Notes:
            This method updates the resolution attribute and recalculates the camera matrix K.
        """
        self.resolution = (resolution[0] * self.ratio, resolution[1] * self.ratio)
        self.K = np.array([[self.focal_length, 0., self.resolution[1] / 2.],
                          [0., self.focal_length, self.resolution[0] / 2.],
                          [0., 0., 1.]])
    
    def __call__(self, verts, faces=None, color=None, color_type='white', R=None, mesh_filename=None,
                img=np.zeros((224, 224, 3)), cam=np.array([1, 0, 0]),
                rgba=False, addlight=True):
        '''
        Render mesh using OpenDR.

        Args:
            verts (np.ndarray): Vertices of the mesh to render. Shape - (V, 3)
            faces (np.ndarray): Faces of the mesh. Shape - (F, 3)
            color (tuple or np.ndarray): RGB color tuple or array.
            color_type (str): Type of predefined color to use if color is not provided.
            R (np.ndarray): Rotation matrix to manipulate verts. Shape - [3, 3]
            mesh_filename (str): Optional filename of the mesh.
            img (np.ndarray): Background image to overlay the rendered mesh on. Shape - (224, 224, 3)
            cam (np.ndarray): Camera parameters.
            rgba (bool): Whether to render with alpha channel.
            addlight (bool): Whether to add lighting effects.

        Returns:
            np.ndarray: Rendered image(s) with shape (224, 224, 3) or (224, 224, 4) if rgba=True.
        '''
        ## Create OpenDR renderer
        rn = self.renderer
        h, w = self.resolution
        K = self.K

        f = np.array([K[0, 0], K[1, 1]])
        c = np.array([K[0, 2], K[1, 2]])
        
        if faces is None:
            faces = self.faces
        if len(cam) == 4:
            t = np.array([cam[2], cam[3], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
        elif len(cam) == 3:
            t = np.array([cam[1], cam[2], 2 * K[0, 0] / (w * cam[0] + 1e-9)])
    
        rn.camera = ProjectPoints(rt=np.array([0, 0, 0]), t=t, f=f, c=c, k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 1000., 'width': w, 'height': h}

        albedo = np.ones_like(verts)*.9

        if color is not None:
            color0 = np.array(color)
            color1 = np.array(color)
            color2 = np.array(color)
        elif color_type == 'white':
            color0 = np.array([1., 1., 1.])
            color1 = np.array([1., 1., 1.])
            color2 = np.array([0.7, 0.7, 0.7])
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]
        else:
            color0 = self.colors_dict[color_type] * 1.2
            color1 = self.colors_dict[color_type] * 1.2
            color2 = self.colors_dict[color_type] * 1.2
            color = np.ones_like(verts) * self.colors_dict[color_type][None, :]

        if R is not None:
            assert R.shape == (3, 3), "Shape of rotation matrix should be (3, 3)"
            verts = np.dot(verts, R)

        rn.set(v=verts, f=faces, vc=color, bgcolor=np.zeros(3))

        if addlight:
            yrot = np.radians(120)  # Angle of lights

            # First light source
            rn.vc = LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-200, -100, -100]), yrot),
                vc=albedo,
                light_color=color0)

            # Second light source
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([800, 10, 300]), yrot),
                vc=albedo,
                light_color=color1)

            # Third light source
            rn.vc += LambertianPointLight(
                f=rn.f,
                v=rn.v,
                num_verts=len(rn.v),
                light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
                vc=albedo,
                light_color=color2)

        rendered_image = rn.r
        visibility_image = rn.visibility_image

        image_list = [img] if type(img) is not list else img

        return_img = []
        for item in image_list:
            if self.ratio != 1:
                img_resized = resize(item, (item.shape[0] * self.ratio, item.shape[1] * self.ratio), anti_aliasing=True)
            else:
                img_resized = item / 255.

            try:
                img_resized[visibility_image != (2**32 - 1)] = rendered_image[visibility_image != (2**32 - 1)]
            except:
                logger.warning('Can not render mesh.')

            img_resized = (img_resized * 255).astype(np.uint8)
            res = img_resized

            if rgba:
                img_resized_rgba = np.zeros((img_resized.shape[0], img_resized.shape[1], 4))
                img_resized_rgba[:, :, :3] = img_resized
                img_resized_rgba[:, :, 3][visibility_image != (2**32 - 1)] = 255
                res = img_resized_rgba.astype(np.uint8)
            return_img.append(res)

        if type(img) is not list:
            return_img = return_img[0]

        return return_img


def rotateY(points, angle):
    """
    Rotate all points in a 2D array around the y axis.

    Args:
    - points: numpy array of shape (n, 3) where n is the number of points
    - angle: angle in radians by which to rotate the points around the y axis

    Returns:
    - rotated_points: numpy array of shape (n, 3) containing rotated points
    """
    # Rotation matrix around y-axis
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    # Apply rotation using dot product
    rotated_points = np.dot(points, ry)
    return rotated_points

def rotateX(points, angle):
    """
    Rotate all points in a 2D array around the x axis.

    Args:
    - points: numpy array of shape (n, 3) where n is the number of points
    - angle: angle in radians by which to rotate the points around the x axis

    Returns:
    - rotated_points: numpy array of shape (n, 3) containing rotated points
    """
    # Rotation matrix around x-axis
    rx = np.array([
        [1., 0., 0.],
        [0., np.cos(angle), -np.sin(angle)],
        [0., np.sin(angle), np.cos(angle)]
    ])
    # Apply rotation using dot product
    rotated_points = np.dot(points, rx)
    return rotated_points

def rotateZ(points, angle):
    """
    Rotate all points in a 2D array around the z axis.

    Args:
    - points: numpy array of shape (n, 3) where n is the number of points
    - angle: angle in radians by which to rotate the points around the z axis

    Returns:
    - rotated_points: numpy array of shape (n, 3) containing rotated points
    """
    # Rotation matrix around z-axis
    rz = np.array([
        [np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle), 0.],
        [0., 0., 1.]
    ])
    # Apply rotation using dot product
    rotated_points = np.dot(points, rz)
    return rotated_points


class IUV_Renderer(object):
    def __init__(self, focal_length=5000., orig_size=224, output_size=56, mode='iuv', device=torch.device('cuda'), mesh_type='smpl'):
        """
        Initialize the renderer object.

        Parameters:
        - focal_length: float, focal length of the camera
        - orig_size: int, original size of the image
        - output_size: int, output size of the rendered image
        - mode: str, mode of operation ('iuv', 'pncc', 'seg')
        - device: torch.device, device to run computations on
        - mesh_type: str, type of mesh ('smpl' or other)

        Initializes various attributes based on the mode of operation.
        """
        self.focal_length = focal_length
        self.orig_size = orig_size
        self.output_size = output_size

        if mode == 'iuv':
            if mesh_type == 'smpl':
                # Initialize DensePoseMethods and load necessary data
                DP = DensePoseMethods()

                # Vertex mapping and faces from DensePose
                vert_mapping = DP.All_vertices.astype('int64') - 1
                self.vert_mapping = torch.from_numpy(vert_mapping)
                faces = DP.FacesDensePose
                faces = faces[None, :, :]
                self.faces = torch.from_numpy(faces.astype(np.int32))     # [1, 13774, 3], torch.int32

                # Number of parts
                num_part = float(np.max(DP.FaceIndices))
                self.num_part = num_part

                # Load or create vertex part IDs
                dp_vert_pid_fname = 'data/dp_vert_pid.npy'
                if os.path.exists(dp_vert_pid_fname):
                    dp_vert_pid = list(np.load(dp_vert_pid_fname))
                else:
                    print('creating data/dp_vert_pid.npy')
                    dp_vert_pid = []
                    for v in range(len(vert_mapping)):
                        for i, f in enumerate(DP.FacesDensePose):
                            if v in f:
                                dp_vert_pid.append(DP.FaceIndices[i])
                                break
                    np.save(dp_vert_pid_fname, np.array(dp_vert_pid))

                # Texture vertices
                textures_vts = np.array(
                    [(dp_vert_pid[i] / num_part, DP.U_norm[i], DP.V_norm[i]) for i in range(len(vert_mapping))]
                )
                self.textures_vts = torch.from_numpy(textures_vts[None].astype(np.float32))   # (1, 7829, 3)

        elif mode == 'pncc':
            # For PNCC mode
            self.vert_mapping = None
            self.faces = torch.from_numpy(get_model_faces(mesh_type)[None].astype(np.int32))    #  mano: torch.Size([1, 1538, 3])
            textures_vts = get_model_tpose(mesh_type).unsqueeze(0)   # mano: torch.Size([1, 778, 3])

            # Normalize textures_vts
            texture_min = torch.min(textures_vts) - 0.001
            texture_range = torch.max(textures_vts) - texture_min + 0.001
            self.textures_vts = (textures_vts - texture_min) / texture_range

        elif mode == 'seg':
            # For segmentation mode
            self.vert_mapping = None
            body_model = 'smpl'

            # Load faces and vertex segmentation
            self.faces = torch.from_numpy(get_smpl_faces().astype(np.int32)[None])
            with open(os.path.join(path_config.SMPL_MODEL_DIR, '{}_vert_segmentation.json'.format(body_model)), 'rb') as json_file:
                smpl_part_id = json.load(json_file)

            # Create vertex ID mapping
            v_id = []
            for k in smpl_part_id.keys():
                v_id.extend(smpl_part_id[k])
            v_id = torch.tensor(v_id)
            n_verts = len(torch.unique(v_id))
            num_part = len(constants.SMPL_PART_ID.keys())
            self.num_part = num_part

            # Create segmentation texture vertices
            seg_vert_pid = np.zeros(n_verts)
            for k in smpl_part_id.keys():
                seg_vert_pid[smpl_part_id[k]] = constants.SMPL_PART_ID[k]
            textures_vts = seg_vert_pid[:, None].repeat(3, axis=1) / num_part
            self.textures_vts = torch.from_numpy(textures_vts[None].astype(np.float32))

        # Camera intrinsics
        K = np.array([[self.focal_length, 0., self.orig_size / 2.],
                      [0., self.focal_length, self.orig_size / 2.],
                      [0., 0., 1.]])

        # Rotation matrix and translation vector
        R = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        t = np.array([0, 0, 5])

        # Adjust camera intrinsics if original size is not 224
        if self.orig_size != 224:
            rander_scale = self.orig_size / float(224)
            K[0, 0] *= rander_scale
            K[1, 1] *= rander_scale
            K[0, 2] *= rander_scale
            K[1, 2] *= rander_scale

        # Store camera intrinsics as PyTorch tensors
        self.K = torch.FloatTensor(K[None, :, :])
        self.R = torch.FloatTensor(R[None, :, :])
        self.t = torch.FloatTensor(t[None, None, :])

        # Adjusted camera matrix for rendering
        camK = F.pad(self.K, (0, 1, 0, 1), "constant", 0)
        camK[:, 2, 2] = 0
        camK[:, 3, 2] = 1
        camK[:, 2, 3] = 1
        self.K = camK

        # Device to run computations on
        self.device = device

        # Ambient lights for rendering
        lights = AmbientLights(device=self.device)

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=output_size,
            blur_radius=0,
            faces_per_pixel=1,
        )

        # Mesh renderer setup
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(
                device=self.device,
                lights=lights,
                blend_params=BlendParams(background_color=[0, 0, 0], sigma=0.0, gamma=0.0)
            )
        )

    def camera_matrix(self, cam):
        """
        Compute camera matrices.

        Parameters:
        - cam: tensor, camera parameters

        Returns:
        - K: tensor, camera intrinsic matrix
        - R: tensor, rotation matrix
        - t: tensor, translation vector

        Computes camera matrices based on input camera parameters.
        """
        batch_size = cam.size(0)
        K = self.K.repeat(batch_size, 1, 1)
        R = self.R.repeat(batch_size, 1, 1)
        t = torch.stack([-cam[:, 1], -cam[:, 2], 2 * self.focal_length/(self.orig_size * cam[:, 0] + 1e-9)], dim=-1)

        if cam.is_cuda:
            K = K.to(cam.device)
            R = R.to(cam.device)
            t = t.to(cam.device)

        return K, R, t

    def verts2iuvimg(self, verts, cam, iwp_mode=True):
        """
        Render IUV image from vertices.

        Parameters:
        - verts: tensor, vertex positions
        - cam: tensor, camera parameters
        - iwp_mode: bool, whether to use IWP mode

        Returns:
        - iuv_image: tensor, rendered IUV image

        Renders IUV image from input vertices and camera parameters.
        """
        batch_size = verts.size(0)

        # Compute camera matrices
        K, R, t = self.camera_matrix(cam)

        # Select vertices based on mapping or use original vertices
        if self.vert_mapping is None:
            vertices = verts
        else:
            vertices = verts[:, self.vert_mapping, :]

        # Create Meshes object
        mesh = Meshes(vertices, self.faces.to(verts.device).expand(batch_size, -1, -1))

        # Assign texture vertices
        mesh.textures = TexturesVertex(verts_features=self.textures_vts.to(verts.device).expand(batch_size, -1, -1))

        # Create PerspectiveCameras object
        cameras = PerspectiveCameras(device=verts.device, R=R, T=t, K=K, in_ndc=False, image_size=[(self.orig_size, self.orig_size)])

        # Render IUV image
        iuv_image = self.renderer(mesh, cameras=cameras)
        iuv_image = iuv_image[..., :3].permute(0, 3, 1, 2)

        return iuv_image
