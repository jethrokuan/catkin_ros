import cv2
import torch
import numpy as np
from skimage.filters import gaussian
try:
    from ggrasp.utils.timeit import TimeIt
except ImportError:
    try:
        from dougsm_helpers.timeit import TimeIt
    except ImportError:
        print("Can't import timeit module")

def post_process_output(q_img, cos_img, sin_img, width_img, filters=(2.0, 1.0, 1.0)):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    if filters[0]:
        q_img = gaussian(q_img, filters[0], preserve_range=True)
    if filters[1]:
        ang_img = gaussian(ang_img, filters[1], preserve_range=True)
    if filters[2]:
        width_img = gaussian(width_img, filters[2], preserve_range=True)

    q_img = np.clip(q_img, 0.0, 1.0-1e-3)

    return q_img, ang_img, width_img

def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape

    with TimeIt('1'):
        # Crop.
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    with TimeIt('2'):
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    with TimeIt('3'):
        depth_crop[depth_nan_mask==1] = 0

    with TimeIt('4'):
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

        with TimeIt('Inpainting'):
            depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

    with TimeIt('5'):
        # Resize
        depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        with TimeIt('6'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop


def predict(depth, model, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)

    # Inference
    depth = np.clip((depth - depth.mean()), -1, 1)
    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(depth.get_device())
    with torch.no_grad():
        pos_out, cos_out, sin_out, width_out = model(depthT)

    # pos_out[depth_nan_mask] = 0
    q_img, ang_img, width_img = post_process_output(pos_out, cos_out, sin_out, width_out, filters)

    return q_img, ang_img, width_img, depth.squeeze()
