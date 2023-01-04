import os
import random
import re
import time
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor, sigmoid
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision.utils import make_grid
from sklearn import metrics
import math 
from typing import *
from collections import defaultdict
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution, Bernoulli, Normal
import torchvision.utils as vutils
from torch.nn.functional import softplus
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from umap import UMAP
from matplotlib import pyplot as plt
import seaborn as sns
#import plotly.express as px
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from model_VAE_plus import SingleCellDataset, PrintSize, Flatten, UnFlatten
from model_VAE_plus import ReparameterizedDiagonalGaussian
from model_VAE_plus import ReparameterizedDiagonalGaussianWithSigmoid
from model_VAE_plus import VariationalAutoencoder, VariationalInference

print(torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

print("PyTorch Version {}" .format(torch.__version__))
print("Cuda Version {}" .format(torch.version.cuda))
print("CUDNN Version {}" .format(torch.backends.cudnn.version()))

torch.backends.cudnn.enabled = True

result_dir = 'result_visualizations/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

# Set random seed for reproducibility
#manualSeed = 2331
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
f = open(result_dir + 'random_seed.txt', "w")
f.write(str(manualSeed))
f.close()
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Path to the model to be used
vae_model = '/work3/s193518/deep-learning-project-03/code/results_vanilla_vae_1/vae_final.model'
vae_plus_model = '/work3/s193518/deep-learning-project-03/code/results_plus_VAE_1/vae_plus_final.model'

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 68

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector
latent_features = 100

# Size of feature maps in VAE encoder and decoder
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Max patience for early stopping
max_patience = 50

# Learning rate for optimizers
lr = 1e-4

# Beta hyperparam for VAE loss
beta = 1.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The value the DMSO category is downsampled to
downsample_value = 16000

# Amount of data used for training, validation and testing
data_prct = 1
train_prct = 0.95

# Number of classes
n_classes = 13

### Dataset

# Load metadata table
start_time = time.time()
metadata = pd.read_csv('/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv')
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

# DMSO category
DMSO_indx = metadata.index[metadata['moa'] == 'DMSO']
DMSO_drop_indices = np.random.choice(DMSO_indx, size=len(DMSO_indx) - downsample_value, replace=False)

# Microtubule stabilizers
#micro_indx = metadata.index[metadata['moa'] == 'Microtubule stabilizers']
#micro_drop_indices = np.random.choice(micro_indx, size=len(micro_indx) - downsample_value, replace=False)

#print(len(np.intersect1d(DMSO_drop_indices, micro_drop_indices)))
#all_drop_indices = np.concatenate((DMSO_drop_indices, micro_drop_indices))
all_drop_indices = DMSO_drop_indices

metadata_subsampled = metadata.drop(all_drop_indices, axis=0).reset_index()

# Map from class name to class index
classes = {index: name for name, index in enumerate(metadata["moa"].unique())}
classes_inv = {v: k for k, v in classes.items()}

# The loaders perform the actual work
#images_folder = "/Users/mikkelrasmussen/mnt/deep_learning_project/data/singh_cp_pipeline_singlecell_images"
images_folder = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/'
train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x/x.max()),
     #transforms.Lambda(lambda x: x/torch.amax(x, dim=(0, 1))),
     #transforms.Lambda(lambda x: torch.flatten(x))
    ]
)

train_set = SingleCellDataset(images_folder=images_folder, 
                              annotation_file=metadata_subsampled, 
                              transform=train_transforms,
                              class_map=classes)

# Define the size of the train, validation and test datasets
data_amount = int(len(metadata_subsampled) * data_prct)
train_size = int(train_prct * data_amount)
val_size = (data_amount - train_size) // 2
test_size = (data_amount - train_size) // 2

indicies = torch.randperm(len(metadata_subsampled))
train_indices = indicies[:train_size]
val_indicies = indicies[train_size:train_size+val_size]
test_indicies = indicies[train_size+val_size:train_size+val_size+test_size]

training_set = torch.utils.data.Subset(train_set, train_indices.tolist())
val_set = torch.utils.data.Subset(train_set, val_indicies.tolist())
testing_set = torch.utils.data.Subset(train_set, test_indicies.tolist())

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, 
                                             shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

# Load a batch of images into memory
images, labels = next(iter(training_loader))
vae = VariationalAutoencoder(latent_features)
loss_fn = nn.MSELoss(reduction='none')
print(vae)

# Test with random input
vi_test = VariationalInference(beta=1)
print(images.shape)
loss, xhat, diagnostics, outputs = vi_test(vae, images)
print(f"{'loss':6} | mean = {loss:10.3f}, shape: {list(loss.shape)}")
for key, tensor in diagnostics.items():
    print(f"{key:6} | mean = {tensor.mean():10.3f}, shape: {list(tensor.shape)}")

def plot_samples_and_reconstructions(x, px1, px2, fname='latent_samples', nrow=2):
    x = x.to('cpu')
    x = make_grid(x.view(-1, 3, 68, 68), nrow=nrow).permute(1, 2, 0)
    
    px1 = px1.to('cpu')
    px1 = make_grid(px1.view(-1, 3, 68, 68), nrow=nrow).permute(1, 2, 0)

    px2 = px2.to('cpu')
    px2 = make_grid(px2.view(-1, 3, 68, 68), nrow=nrow).permute(1, 2, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 18), squeeze=False)
    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('VAE')
    axes[0, 2].set_title('VAE+')

    axes[0, 0].imshow(x)
    axes[0, 1].imshow(px1)
    axes[0, 2].imshow(px2)

    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    fig.savefig(result_dir + fname + '.png', bbox_inches='tight')

# Evaluator: Variational Inference
vi = VariationalInference(beta=beta)

# Load pre-train VAE model
modelVAE = torch.load(vae_model)
modelVAE = modelVAE.to('cpu')

# Load pre-train VAE+ model
modelVAEplus = torch.load(vae_plus_model)
modelVAEplus = modelVAEplus.to('cpu')

# Load a batch of images into memory
x, y = next(iter(training_loader))

# Pass batch through vanilla VAE
loss_elbo, xhat, diagnostics, vae_outputs = vi(modelVAE, x)
vae_px = vae_outputs['px']
vae_x_sample = vae_px.sample().to('cpu')

# Pass batch through vanilla VAE
loss_elbo, xhat, diagnostics, vae_plus_outputs = vi(modelVAEplus, x)
vae_plus_px = vae_plus_outputs['px']
vae_plus_x_sample = vae_plus_px.sample().to('cpu')

for i in range(10):
    filename = 'real_vs_reconstructions' + str(i)
    plot_samples_and_reconstructions(x[i:i+4], px1=vae_x_sample[i:i+4], 
                                     px2=vae_plus_x_sample[i:i+4], 
                                     fname = filename)


#plot_interpolations(modelVAE)