from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
from AIG_MSPU.training.loss_functions.deep_supervision import MultipleOutputLoss2
from AIG_MSPU.utilities.to_torch import maybe_to_torch, to_cuda
from AIG_MSPU.network_architecture.initialization import InitWeights_He
from AIG_MSPU.network_architecture.neural_network import SegmentationNetwork
from AIG_MSPU.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from AIG_MSPU.training.dataloading.dataset_loading import unpack_dataset
from AIG_MSPU.training.network_training.nnUNetTrainer import nnUNetTrainer
from AIG_MSPU.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from AIG_MSPU.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
import time
from tqdm import tqdm
import SimpleITK as sitk

from AIG_MSPU.training.data_augmentation.data_augmentation_moreDA_probv2 import get_moreDA_augmentation_probv2
from AIG_MSPU.training.dataloading.data_loading_prob import DataLoader3D_prob
from AIG_MSPU.SSL.utils import load_encoder_weights_diffcin
from AIG_MSPU.SSL.run_pretrain import params
from AIG_MSPU.network_architecture.generic_modular_UNet import PlainConvUNet, get_default_network_config

class Trainer_AIG_MSPU(nnUNetTrainer):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

        self.ckpt_path = "/home/usr/Segmentation/SSL/ppm.pth"
        self.ssl_params = params
        self.ssl_mode = 'pixpro'

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                
                self.prob_path = "/nvme1date/usr/nnunet_data/pre/Task517_MONET/prob_maps/prob_fusion.npy"
                self.dl_tr, self.dl_val = self.get_basic_generators_presave_prob()
                
                unpack_dataset(self.folder_with_preprocessed_data)

                print(self.dl_tr.num_dict)
                
                check_train_data = False
                if check_train_data:
                    print('Train data before DA:')
                    data_dict = next(self.dl_tr)
                    print(data_dict.keys())
                    print(data_dict['keys'])
                    print(data_dict['data'].shape, data_dict['data'].dtype)
                    print(data_dict['seg'].shape, data_dict['seg'].dtype)

                check_val_data = False
                if check_val_data:
                    print('VAL data before DA:')
                    data_dict = next(self.dl_val)
                    print(data_dict.keys())
                    print(data_dict['keys'])
                    print(data_dict['data'].shape, data_dict['data'].dtype)
                    print(data_dict['seg'].shape, data_dict['seg'].dtype)
                
                self.tr_gen, self.val_gen = get_moreDA_augmentation_probv2(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )

                check_train_data = False
                if check_train_data:
                    print('Train data after DA:')
                    data_dict = next(self.tr_gen)
                    print(data_dict.keys())
                    # print(data_dict['properties'])
                    print(data_dict['keys'])
                    print(data_dict['data'].shape, data_dict['data'].dtype)
                    for i in data_dict['target']:
                        print(i.shape, i.dtype)
                
                check_val_data = False
                if check_val_data:
                    print('Val data after DA:')
                    data_dict = next(self.val_gen)
                    print(data_dict.keys())
                    print(data_dict['keys'])
                    print(data_dict['data'].shape, data_dict['data'].dtype)
                    for i in data_dict['target']:
                        print(i.shape, i.dtype)
                
                # Check Image
                check_image = False
                if check_image:
                    print('CHECK IMAGE: TRUE')
                    for i in range(self.batch_size):
                        
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['data'][i, 0]), '/date/usr/monet/check/%dimg.nii.gz'%i)
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['target'][0][i, 0]), '/date/usr/monet/check/%dseg.nii.gz'%i)
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['data'][i, 1]), '/date/usr/monet/check/%dprob_1.nii.gz'%i)
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['data'][i, 2]), '/date/usr/monet/check/%dprob_2.nii.gz'%i)
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['data'][i, 3]), '/date/usr/monet/check/%dprob_3.nii.gz'%i)
                        sitk.WriteImage(sitk.GetImageFromArray(data_dict['data'][i, 6]), '/date/usr/monet/check/%dprob_6.nii.gz'%i)
                
                q1 = input('the presave_folder is %s?' % self.presave_folder)
                q3 = input('the prob_path is %s?' % self.prob_path)
                q2 = input('the fold is %d?' % self.fold)
                
                assert q1=='y' and q2=='y' and q3=='y'

                # Speed Test
                speed_test = True
                val_or_train = 'train'
                if speed_test:
                    start = time.time()
                    for i in tqdm(range(0, 100)):
                        if val_or_train == 'train':
                            data_dict = next(self.tr_gen)
                        else:
                            data_dict = next(self.val_gen)
                    print('TOTAL TIME:', time.time() - start)
                
                print('AVAILABLE TEST!')
                _ = self.tr_gen.next()
                _ = self.val_gen.next()
                print('DONE!')
                

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def get_basic_generators_presave_prob(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:

            dl_tr = DataLoader3D_prob(self.prob_file, self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                            False, oversample_foreground_percent=self.oversample_foreground_percent,
                            pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D_prob(self.prob_file, self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        
        return dl_tr, dl_val
    
    def initialize_network(self):
        assert self.threeD, 'This Trainer Only Supports 3D Network Currently'
        cfg = get_default_network_config(3, None, norm_type="in")

        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        blocks_per_stage_encoder = [2, 2, 2, 2, 2, 2]
        blocks_per_stage_decoder = [2, 2, 2, 2, 2]
        pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        
        self.num_input_channels = self.num_input_channels + self.num_classes - 1
        
        self.network = PlainConvUNet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                     pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                     blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))
        print(self.network)
        # ckpt_path = "/home4/usr/2112_MONET/2021-12-22-SSL/result/2022-2-17/ckpt_epoch_200.pth"
        q1 = input('LOADING PIXPRO PRETRAINED MODEL FROM %s?' % self.ckpt_path)
        assert q1 == 'y', 'Please Make Sure Of The Settings!'
        load_encoder_weights_diffcin(self.ssl_mode, self.network, self.ckpt_path, self.ssl_params)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):

        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):

        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:

        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):

        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):

        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):

        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = super().run_training()
        self.network.decoder.deep_supervision = ds
        return ret
