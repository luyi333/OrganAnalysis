from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
import numpy as np
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_spatial_2, \
    augment_channel_translation, \
    augment_mirroring, augment_transpose_axes, augment_zoom, augment_resize, augment_rot90
import numpy as np
from batchgenerators.augmentations.noise_augmentations import augment_blank_square_noise, augment_gaussian_blur, \
    augment_gaussian_noise, augment_rician_noise
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from warnings import warn

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
import numpy as np

import numpy as np
import os
import abc

from collections import OrderedDict

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_cached_per_thread": 1,
}

def GetDefaultDA(patch_size=[144, 144, 144]):
    '''
    Default Data Augmentation Without Spatial Transforms
    '''
    params = default_3D_augmentation_params
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                        p_per_sample=0.1))  # inverted gamma
    if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                            p_per_sample=params["p_gamma"]))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

def nnUNetPadding(data_ori, seg_ori, use_atlas=False, atlas=None, patch_size=(207, 207, 207), final_patch_size=(144, 144, 144), pad_mode="constant"):
    
    data_shape, seg_shape = determine_shapes(use_atlas=use_atlas, patch_size=patch_size)
    data = np.zeros(data_shape, dtype=np.float32)
    seg = np.zeros(seg_shape, dtype=np.float32)
    pad_kwargs_data = OrderedDict()
    force_fg = False

    case_all_data = np.concatenate((np.expand_dims(data_ori, 0), np.expand_dims(seg_ori, 0)), 0)

    need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
    for d in range(3):
        # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
        # always
        if need_to_pad[d] + case_all_data.shape[d + 1] < patch_size[d]:
            need_to_pad[d] = patch_size[d] - case_all_data.shape[d + 1]

    # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
    # define what the upper and lower bound can be to then sample form them with np.random.randint
    shape = case_all_data.shape[1:]
    lb_x = - need_to_pad[0] // 2
    ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - patch_size[0]
    lb_y = - need_to_pad[1] // 2
    ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - patch_size[1]
    lb_z = - need_to_pad[2] // 2
    ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - patch_size[2]

    # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
    # at least one of the foreground classes in the patch
    if not force_fg:
        bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
        bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
        bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
    else:
        # these values should have been precomputed
        if 'class_locations' not in properties.keys():
            raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

        # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
        foreground_classes = np.array(
            [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
        foreground_classes = foreground_classes[foreground_classes > 0]

        if len(foreground_classes) == 0:
            # this only happens if some image does not contain foreground voxels at all
            selected_class = None
            voxels_of_that_class = None
            print('case does not contain any foreground classes', i)
        else:
            selected_class = np.random.choice(foreground_classes)

            voxels_of_that_class = properties['class_locations'][selected_class]

        if voxels_of_that_class is not None:
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
            # Make sure it is within the bounds of lb and ub
            bbox_x_lb = max(lb_x, selected_voxel[0] - patch_size[0] // 2)
            bbox_y_lb = max(lb_y, selected_voxel[1] - patch_size[1] // 2)
            bbox_z_lb = max(lb_z, selected_voxel[2] - patch_size[2] // 2)
        else:
            # If the image does not contain any foreground classes, we fall back to random cropping
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

    bbox_x_ub = bbox_x_lb + patch_size[0]
    bbox_y_ub = bbox_y_lb + patch_size[1]
    bbox_z_ub = bbox_z_lb + patch_size[2]

    # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
    # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
    # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
    # later
    valid_bbox_x_lb = max(0, bbox_x_lb)
    valid_bbox_x_ub = min(shape[0], bbox_x_ub)
    valid_bbox_y_lb = max(0, bbox_y_lb)
    valid_bbox_y_ub = min(shape[1], bbox_y_ub)
    valid_bbox_z_lb = max(0, bbox_z_lb)
    valid_bbox_z_ub = min(shape[2], bbox_z_ub)

    # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
    # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
    # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
    # remove label -1 in the data augmentation but this way it is less error prone)
    case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub,
                            valid_bbox_z_lb:valid_bbox_z_ub])

    data[0] = np.pad(case_all_data[:-1], ((0, 0),
                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                        pad_mode, **pad_kwargs_data)
    # print('case_all_data[-1:].shape:', case_all_data[-1:].shape)
    seg[0, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                        'constant', **{'constant_values': -1})
    if use_atlas:
        # print('atlas.shape:', atlas.shape)
        seg[0, 1:] = np.pad(atlas, ((0, 0),
                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                            (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                            'constant', **{'constant_values': -1})
    
    return data, seg

def determine_shapes(use_atlas=False, patch_size=(144, 144, 144)):
    if use_atlas:
        num_seg = 15
    else:
        num_seg = 1
    data_shape = (1, 1, *patch_size)
    seg_shape = (1, num_seg, *patch_size)
    return data_shape, seg_shape

class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str

class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param keys: specify keys to be converted to tensors. If None then all keys will be converted
        (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
        :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
        and long supported (use string)
        """
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            else:
                raise ValueError('Unknown value for cast_to: %s' % self.cast_to)
        return tensor

    def __call__(self, data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(val)).contiguous()
                elif isinstance(val, (list, tuple)) and all([isinstance(i, np.ndarray) for i in val]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in val]
        else:
            for key in self.keys:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(data_dict[key])).contiguous()
                elif isinstance(data_dict[key], (list, tuple)) and all([isinstance(i, np.ndarray) for i in data_dict[key]]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in data_dict[key]]

        return data_dict

class RenameTransform(AbstractTransform):
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Optionally removes data_dict[in_key] from the dict.
    '''

    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict

class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1, p_independent_scale_per_axis: int=1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis, 
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict

def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result

class Compose(AbstractTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"

class GaussianNoiseTransform(AbstractTransform):
    """Adds additive Gaussian Noise

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """

    def __init__(self, noise_variance=(0, 0.1), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.noise_variance = noise_variance

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b], self.noise_variance)
        return data_dict

class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma=(1, 5), data_key="data", label_key="seg", different_sigma_per_channel=True,
                 p_per_channel=1, p_per_sample=1):
        """

        :param blur_sigma:
        :param data_key:
        :param label_key:
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.label_key = label_key
        self.blur_sigma = blur_sigma

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
                                                                 self.different_sigma_per_channel, self.p_per_channel)
        return data_dict

class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
                                                                                self.multiplier_range,
                                                                                self.per_channel)
        return data_dict

class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, data_key="data",
                 p_per_sample=1):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
                                                               contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel)
        return data_dict

class SimulateLowResolutionTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_linear_downsampling_scipy(data_dict[self.data_key][b],
                                                                                zoom_range=self.zoom_range,
                                                                                per_channel=self.per_channel,
                                                                                p_per_channel=self.p_per_channel,
                                                                                channels=self.channels,
                                                                                order_downsample=self.order_downsample,
                                                                                order_upsample=self.order_upsample,
                                                                                ignore_axes=self.ignore_axes)
        return data_dict

class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict

class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            ret_val = augment_mirroring(data[b], sample_seg, axes=self.axes)
            data[b] = ret_val[0]
            if seg is not None:
                seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict

class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict