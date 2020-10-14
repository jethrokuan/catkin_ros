import glob
import os
import random

import numpy as np
import torch
import torch.utils.data

import matplotlib.pyplot as plt

from ggrasp.utils.dataset_processing import grasp, image

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self,
                 width=640,
                 height=480,
                 output_size=224,
                 include_depth=True,
                 include_rgb=True
                 ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_depth(self, img):
        depth_img = image.Image(img)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        depth_img.normalise()
        # depth_img.resize((self.output_size, self.output_size))
        depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img

    def get_rgb(self, img, norm=True):
        rgb_img = image.Image(img)
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        # rgb_img.resize((self.output_size, self.output_size))
        if norm:
                rgb_img.normalise()
                rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_data(self, rgb=None, depth=None):
        depth_img = None
        rgb_img = None
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(img=depth)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         np.expand_dims(rgb_img, 0)),
                        1
                    )
                )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(np.expand_dims(rgb_img, 0))

        return x, depth_img, rgb_img


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', 'r.png') for f in self.depth_files]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img


class JacquardDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """

    def __init__(self, file_path, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', '*_grasps.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('perfect_depth.tiff', 'RGB.png') for f in self.depth_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])

class CustomDataset(GraspDatasetBase):
    """
    Dataset wrapper for the custom collected dataset.
    """

    def __init__(self, file_path, **kwargs):
        """
        :param file_path: Dataset directory.
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CustomDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*.grasp'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('.grasp', '.png') for f in self.grasp_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_custom_file(self.grasp_files[idx], scale=1.0)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.Image.from_file(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def plot(self, idx):
        gtbbs = self.get_gtbb(idx)
        depth_img = image.Image.from_file(self.depth_files[idx]).img
        f, ax = plt.subplots()
        ax.imshow(depth_img)
        gtbbs.show(ax)
        return f
