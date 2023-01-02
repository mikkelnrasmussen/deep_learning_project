import os
import time
import re
import random
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
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
from torch.distributions import Distribution, Bernoulli
import torchvision.utils as vutils
from torch.nn.functional import softplus
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
from sklearn import metrics
from model_VAE_plus import SingleCellDataset, PrintSize, Flatten, UnFlatten
from model_VAE_plus import ReparameterizedDiagonalGaussian
from model_VAE_plus import ReparameterizedDiagonalGaussianWithSigmoid
from model_VAE_plus import VariationalAutoencoder, VariationalInference
from sklearn.ensemble import RandomForestClassifier
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
#sns.set_style("whitegrid")

name = 'classifier_plus_VAE_test'
result_dir = 'classifier_results_plus_VAE_test/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)

# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
f = open(result_dir + 'random_seed.txt', "w")
f.write(str(manualSeed))
f.close()
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# Path to the model to be used
vae_model = '/work3/s193518/deep-learning-project-03/code/results_plus_VAE_3/vae_plus_final.model'
# Initialize parameters
# Number of workers for dataloader
workers = 1

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

# Validation frequency
validation_every_steps = 1000

# Number of training epochs
num_epochs = 5

# Max patience for early stopping
max_patience = 30

# Learning rate for optimizers
lr = 3e-4

# Beta hyperparam for VAE loss
beta = 1.0

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The value the DMSO category is downsampled to
downsample_value = 16000

# Amount of data used for training, validation and testing
data_prct = 0.1
train_prct = 0.85

# Number of classes
n_classes = 13

start_time = time.time()
metadata = pd.read_csv('/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv')
print("pd.read_csv wiht pyarrow took %s seconds" % (time.time() - start_time))

DMSO_indx = metadata.index[metadata['moa'] == 'DMSO']
DMSO_drop_indices = np.random.choice(DMSO_indx, size=260360, replace=False)

# Subsample metadata dataframe
metadata_subsampled = metadata.drop(DMSO_drop_indices).reset_index()
metadata_subsampled.groupby("moa").size().reset_index(name='counts').sort_values(by="counts", ascending=False)

# Shuffle metadata dataframe
metadata_subsampled = metadata_subsampled.sample(frac=1).reset_index(drop=True)

# Map from class name to class index
classes = {index: name for name, index in enumerate(metadata["moa"].unique())}
classes_inv = {v: k for k, v in classes.items()}
classes

# The loaders perform the actual work
images_folder = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/"
train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Lambda(lambda x: torch.flatten(x)),
     transforms.Lambda(lambda x: x/x.max())]
)

train_set = SingleCellDataset(images_folder=images_folder, 
                              annotation_file=metadata_subsampled, 
                              transform=train_transforms,
                              class_map=classes)


data_amount = int(len(metadata_subsampled) * data_prct)
train_size = int(train_prct * data_amount)
#val_size = (data_amount - train_size) // 2
test_size = (data_amount - train_size)

indicies = torch.randperm(len(metadata_subsampled))
train_indices = indicies[:train_size]
#val_indicies = indicies[train_size:train_size+val_size]
test_indicies = indicies[train_size:train_size+test_size]

# Checking there are not overlapping incdicies
#print(sum(np.isin(train_indices.numpy() , [val_indicies.numpy(), test_indicies.numpy()])))
#print(sum(np.isin(val_indicies.numpy() , [train_indices.numpy(), test_indicies.numpy()])))
#print(sum(np.isin(test_indicies.numpy() , [train_indices.numpy(), val_indicies.numpy()])))

training_set = torch.utils.data.Subset(train_set, train_indices.tolist())
#val_set = torch.utils.data.Subset(train_set, val_indicies.tolist())
testing_set = torch.utils.data.Subset(train_set, test_indicies.tolist())

""" training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, 
                                              shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, 
                                          shuffle=True, drop_last=True) """

""" print(len(training_loader.dataset))
print(len(val_loader.dataset))
print(len(test_loader.dataset)) """

images, labels = next(iter(training_loader))

def load_data(data_dir="/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images/"):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x/x.max())]
     )

    trainset = training_set

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


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

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        activation_fn = nn.ReLU
        
        """ self.net = nn.Sequential(
          
            nn.Linear(latent_features, 512),
            activation_fn(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            activation_fn(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            activation_fn(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            activation_fn(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 100),
            activation_fn(),
            nn.Linear(100, n_classes)
            #nn.Flatten()
        ) """

        self.net = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(latent_features, 512),
            activation_fn(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1024),
            activation_fn(),
            nn.Linear(1024, 300),
            activation_fn(),
            nn.Dropout(p=0.1),
            nn.Linear(300, 100),
            activation_fn(),
            nn.Linear(100, n_classes)
            #nn.Flatten()
        )
        
    def forward(self, x):
        return self.net(x)
         
        
classifierNet = Classifier(n_classes)
print(classifierNet)
classifierNet = classifierNet.to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(classifierNet.parameters(), lr=lr)  

latentNet = torch.load(vae_model)
latentNet = latentNet.to(device)

# Evaluator: Variational Inference
vi = VariationalInference(beta=beta)

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(), 
        pred.detach().cpu().numpy(),
        normalize=normalize
    )

def normalize(matrix, axis):
    axis = {'true': 1, 'pred': 0}[axis]
    return matrix / matrix.sum(axis=axis, keepdims=True)

step = 0
classifierNet.train()

train_loss = []
val_loss = []
test_acc, test_loss = [], []
train_accuracies = []
valid_accuracies = []
loss_val = 1000000-1
best_nll = 1000000
patience = 0
        
for epoch in range(num_epochs):
    
    train_accuracies_batches = []
    train_loss_batches = []
    
    for inputs, targets in training_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass, compute gradients, perform one training step.
        loss, xhat, diagnostics, vae_outputs = vi(latentNet, inputs)
        z = vae_outputs['z']
        z = z.to(device)

        # Forward pass.
        output = classifierNet(z)
        output = output.to(device)
        output = output.reshape(-1, 13)
        
        # Compute loss.
        loss = criterion(output, targets)
        train_loss_batches.append(loss.detach().cpu().numpy())
        
        # Clean up gradients from the model.
        optimizer.zero_grad()
        
        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()
        
        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()
        
        # Increment step counter
        step += 1
        
        # Compute accuracy.
        predictions = output.max(1)[1]
        train_accuracies_batches.append(accuracy(targets, predictions))
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            train_loss.append(np.mean(train_loss_batches))
            
            train_accuracies_batches = []
            train_loss_batches = []
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            valid_loss_batches = []

            with torch.no_grad():
                classifierNet.eval()
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    loss_vae_val, xhat, diagnostics, vae_outputs = vi(latentNet, inputs)
                    z = vae_outputs['z']
                    z = z.to(device)

                    output = classifierNet(z)
                    output = output.reshape(-1,13)

                    loss_val = criterion(output, targets)
                    valid_loss_batches.append(loss_val.detach().cpu().numpy())

                    predictions = output.max(1)[1]

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

                classifierNet.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(np.sum(valid_accuracies_batches) / len(testing_set))
            val_loss.append(np.sum(valid_loss_batches) / len(testing_set))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            print(f"                 validation accuracy: {valid_accuracies[-1]}")


    if loss_val < best_nll:
        print('saved!')
        torch.save(classifierNet, result_dir + name + '.model')
        best_nll = loss_val
        patience = 0
    else:
        patience = patience + 1
            
    if patience > max_patience:
        print("Max patience reached! Performing early stopping!")
        break

    

print("Finished training.")

print('saved final model!')
torch.save(classifierNet, result_dir + name + '_final.model')

# Evaluate test set
test_true, test_pred = np.array([]), np.array([])
confusion_matrix = np.zeros((n_classes, n_classes))
best_classifierNet = torch.load(result_dir + name + '.model')
best_classifierNet = best_classifierNet.to(device)
with torch.no_grad():
    classifierNet.eval()
    test_accuracies = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        loss_vae_test, xhat, diagnostics, vae_outputs = vi(latentNet, inputs)
        z = vae_outputs['z']
        z = z.to(device)

        output = best_classifierNet(z)
        output = output.reshape(-1,13)

        loss = criterion(output, targets)

        predictions = output.max(1)[1]

        # Multiply by len(inputs) because the final batch of DataLoader may be smaller (drop_last=True).
        test_accuracies.append(accuracy(targets, predictions) * len(inputs))
        test_true = np.append(test_true, targets.detach().cpu().numpy())
        test_pred = np.append(test_pred, predictions.detach().cpu().numpy())

    confusion_matrix = metrics.confusion_matrix(
        test_true, 
        test_pred,
        normalize=None)

    test_accuracy = np.sum(test_accuracies) / len(testing_set)
    
    classifierNet.train()

print(f"Test accuracy: {test_accuracy:.3f}")
f = open(result_dir + name + '_test_accuracy.txt', "w")
f.write(str(test_accuracy))
f.close()

x_labels = [classes[i] for i in classes]
y_labels = x_labels
plt.figure(figsize=(6, 6))
sns.heatmap(
    ax=plt.gca(),
    data=normalize(confusion_matrix, 'true'),
    annot=True,
    linewidths=0.5,
    cmap="Reds",
    cbar=False,
    fmt=".2f",
    xticklabels=x_labels,
    yticklabels=y_labels,
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.tight_layout()
plt.savefig(result_dir + 'confusion_matrix.png', bbox_inches='tight')
plt.close()

with sns.axes_style('whitegrid'):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=x_labels, y=np.diag(normalize(confusion_matrix, 'true')))
    plt.xticks(rotation=90)
    plt.title("Per-class accuracy")
    plt.ylabel("Accuracy")
    plt.savefig(result_dir + 'per_class_accuracy.png', bbox_inches='tight')
    plt.close()

plt.figure()
#plt.plot(range(50,step, 50), train_accuracies, 'r', range(50,step, 50), valid_accuracies, 'b')
plt.plot(train_accuracies, label="Train Accucary")
plt.legend()
plt.xlabel('steps'), plt.ylabel('Acc')
plt.savefig(result_dir + 'classification_train_accuracy.png')
plt.close()

plt.figure()
#plt.plot(range(50,step, 50), train_accuracies, 'r', range(50,step, 50), valid_accuracies, 'b')
plt.plot(valid_accuracies, label="Validation Accucary")
plt.legend()
plt.xlabel('steps'), plt.ylabel('Acc')
plt.savefig(result_dir + 'classification_val_accuracy.png')
plt.close()

plt.figure(figsize=(10,5))
plt.title("Training loss during training")
plt.plot(train_loss, label="Train loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'training_loss.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.title("Validation loss during training")
plt.plot(val_loss, label="Validation loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(result_dir + 'validation_loss.png', bbox_inches='tight')
plt.close()

np.save(result_dir + 'train_accuracies.npy', train_accuracies)
np.save(result_dir +'valid_accuracies.npy',  valid_accuracies)
np.save(result_dir + 'train_loss.npy', train_loss)
np.save(result_dir +'val_loss.npy',  val_loss)
torch.save(classifierNet, result_dir + name + '.model')



