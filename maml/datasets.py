import os
import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.snn_models import ModelSNN
from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'doublenmnistsequence':
        import numpy as np
        from torchneuromorphic.doublenmnist_torchmeta.doublenmnist_dataloaders import (DoubleNMNIST,
                                                                                       Compose,
                                                                                       ClassNMNISTDataset,
                                                                                       CropDims,
                                                                                       Downsample,
                                                                                       ToCountFrame,
                                                                                       ToTensor)

        data_dir = os.path.join(os.path.expanduser(folder), 'nmnist/n_mnist.hdf5')
        
        ds=2
        size = [2, 32//ds, 32//ds]
            
        chunk_size = 100
        dt = 1
        transform = Compose([
            CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
            Downsample(factor=[dt,1,ds,ds]),
            ToCountFrame(T = chunk_size, size = size),
            ToTensor()])

        target_transform = Categorical(num_ways)
                
        meta_train_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                        meta_train=True,
                                                        transform = transform,
                                                        target_transform = target_transform,
                                                        chunk_size=chunk_size,
                                                        num_classes_per_task=num_ways), 
                                           num_train_per_class = num_shots, 
                                           num_test_per_class = num_shots_test)
        
        meta_val_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                      meta_val=True,
                                                      transform = transform,
                                                      target_transform = target_transform,
                                                      chunk_size=chunk_size,
                                                      num_classes_per_task=num_ways),
                                         num_train_per_class = num_shots,
                                         num_test_per_class = num_shots_test)
        
        meta_test_dataset = ClassSplitter(DoubleNMNIST(root = data_dir,
                                                       meta_test=True,
                                                       transform = transform,
                                                       target_transform = target_transform,
                                                       chunk_size=chunk_size,
                                                       num_classes_per_task=num_ways), 
                                          num_train_per_class = num_shots, 
                                          num_test_per_class = num_shots_test)


        model = ModelSNN(np.prod(size), num_ways, hidden_sizes=[200])
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
