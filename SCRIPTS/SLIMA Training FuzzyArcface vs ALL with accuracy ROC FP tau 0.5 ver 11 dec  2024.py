#!/usr/bin/env python
# coding: utf-8

# TWO ADVANTAGES OF FUZZYARCFACE OVER OTHER ANGULAR ALGORITHMS

# 1. Adaptive Margin Control
# In most loss functions like ArcFace or SphereFace, the angular margin is either static (fixed for all samples) or is class-specific (same for all samples within a class). However, FuzzyArcFace introduces the concept of a fuzzy membership function to dynamically adjust the margin for each sample. This is a significant innovation, as it allows the network to adjust its decision boundary on a per-sample basis.
# 
# Key Points of Adaptive Margin Control:
# Per-Sample Margin Adjustment: Traditional margin-based loss functions (e.g., ArcFace) apply a fixed margin to all samples, regardless of the confidence or quality of the input data. This approach can lead to misclassifications for samples that are difficult or noisy because they are treated the same way as easy-to-classify samples.
# 
# FuzzyArcFace, on the other hand, applies a fuzzy membership function to each sample, which calculates the confidence or quality of the sample based on its similarity to the class prototype. For highly confident samples (e.g., clear, easy-to-recognize faces), the margin is kept high, ensuring strong intra-class compactness. For uncertain or noisy samples (e.g., occluded or blurry faces), the margin is reduced, allowing the model more flexibility in classifying these harder cases.
# Dynamic Adjustment of Decision Boundaries: The fuzzy membership function essentially adjusts the decision boundary dynamically. If the model is unsure about a particular sample (e.g., a face that appears very different from the class prototype due to occlusion), it won’t apply the same strict margin, thereby reducing the risk of misclassification. Conversely, for very clear samples, the model can apply a larger margin, pushing the boundaries further and ensuring greater separation from other classes.
# 
# This leads to adaptive decision boundaries, where the model can fine-tune how much it “trusts” a sample's similarity to its class, thereby avoiding over-confident mistakes that can happen in traditional models with fixed margins.
# Example:
# Consider two face images:
# 
# Image A: A high-quality, front-facing, well-lit face of a person who has been seen many times in training.
# Image B: A low-quality, side-facing, poorly-lit face of the same person, possibly occluded by sunglasses.
# In traditional ArcFace or SphereFace, both Image A and Image B would have the same angular margin applied. This can lead to errors, as Image B is much harder to classify and may not fit neatly within the margin's boundary.
# 
# With FuzzyArcFace, the margin for Image B would be dynamically reduced, allowing more flexibility in how the model interprets this sample. For Image A, a larger margin can be applied, ensuring tighter clustering of confident samples. This adaptive control ensures that the model can handle both easy and difficult samples more effectively.
# 
# Advantages Over Other Methods:
# ArcFace: The margin is fixed for all samples, which can be too strict for hard samples and too loose for easy samples.
# SphereFace2: Although SphereFace2 applies a multiplicative margin, it still treats all samples within a class the same. FuzzyArcFace provides finer control by adjusting the margin on a per-sample basis.
# AdaptiveFace: While AdaptiveFace adjusts the margin based on class difficulty (e.g., smaller classes get larger margins), it doesn’t adapt to individual sample quality within the class. FuzzyArcFace does both.
# 

# 2. Handling Imbalanced Data
# Handling class imbalances in training datasets is one of the most significant challenges in deep learning. In many face recognition datasets, certain identities have far fewer images (underrepresented classes), while others have thousands of images (overrepresented classes). This imbalance can lead to overfitting or poor generalization, especially for underrepresented classes.
# 
# Key Points on Handling Imbalanced Data:
# Smaller Margins for Underrepresented Classes: In scenarios where a class has very few samples (e.g., a person appears in only a few images in the training set), it becomes harder for the model to generalize and create a well-defined feature space for that class. Most traditional loss functions (e.g., ArcFace) apply the same margin to all classes, which can be problematic for underrepresented classes.
# 
# FuzzyArcFace addresses this by adjusting the margin based on the fuzzy membership for each sample. If the model is less confident about a sample from an underrepresented class (e.g., a class with only a few samples), the fuzzy membership function reduces the margin, allowing the model more flexibility in correctly classifying these samples. This prevents the model from forcing underrepresented classes to fit into a rigid decision boundary, which could lead to overfitting on a few examples.
# Prevents Overfitting to Outliers: In imbalanced datasets, there is a risk of overfitting to outlier examples in underrepresented classes. FuzzyArcFace, by reducing the margin for underrepresented or uncertain samples, prevents the model from creating too narrow a decision boundary that is overly influenced by noisy or unusual samples.
# 
# For example, if a class only has 5 images and 1 of them is an extreme outlier (e.g., a blurred image), a fixed-margin loss function may overfit to this outlier. By dynamically reducing the margin for this uncertain sample, FuzzyArcFace can prevent the model from being overly influenced by the outlier.
# Example:
# Consider two classes in a face dataset:
# 
# Class X: An overrepresented class with 1000 images, featuring a variety of angles, lighting, and expressions.
# Class Y: An underrepresented class with only 5 images, all in similar lighting and angle.
# For Class X, the model can learn a well-defined feature space and apply a strict margin to ensure tight intra-class clustering and good separation from other classes. However, for Class Y, applying the same strict margin could force the model to overfit to the limited examples, making it highly sensitive to noise.
# 
# With FuzzyArcFace, the model applies a smaller margin to Class Y, allowing for a more flexible decision boundary that accommodates the limited variability in the available samples. This prevents overfitting to the sparse data of Class Y while still maintaining good separation for the densely populated Class X.
# 
# Advantages Over Other Methods:
# ArcFace: Treats all classes equally, applying the same margin to both overrepresented and underrepresented classes, which can lead to overfitting for smaller classes.
# AdaptiveFace: Adjusts the margin based on class difficulty, but does so at a class level, not a sample level. While it helps underrepresented classes, it doesn’t provide the per-sample flexibility that FuzzyArcFace offers.
# UniFace: Uses feature normalization but doesn’t address the specific needs of underrepresented classes with dynamic margin adjustment.

# Conclusion: Why These Features Make FuzzyArcFace Superior
# The combination of adaptive margin control and its ability to handle imbalanced data gives FuzzyArcFace a significant advantage over other algorithms in real-world face recognition tasks, particularly when the data is noisy or imbalanced. These features allow FuzzyArcFace to:
# 
# Generalize better to difficult or noisy samples without forcing all data to fit into a rigid decision boundary.
# Handle underrepresented classes by applying smaller, dynamic margins, preventing overfitting and improving classification accuracy for these classes.
# In challenging face recognition scenarios, such as surveillance footage, low-quality images, and imbalanced datasets, these features ensure that FuzzyArcFace remains flexible, making it superior in terms of both accuracy and robustness.

# In[1]:


import os
import random
import numpy as np
from datetime import datetime


# In[2]:


from torch.optim import SGD
from torch.utils.data import Dataset
import torch.nn.functional as F


# In[3]:


from PIL import Image
import tarfile
import zipfile
import math


# In[4]:


# Step 1: Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


# In[5]:


import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models


# In[6]:


#!pip install numpy
import numpy as np
print(np.__version__)


# CONSTANTS

# In[7]:


#CONSTANTS
#lfw_dataset_path = 'lfw'
#pairs_file = 'lfw_test_pair_short.txt'
#lfw_path = '/home/rapids/notebooks/data/storage/slima/lfw/lfw'

# Common directory path
#base_path = '/home/rapids/notebooks/data/storage/slima/DATABASES'

base_path = '/home/rapids/notebooks/slima/DATABASES'
model_save_path = '/home/rapids/notebooks/slima'

# Number of epochs to train for
num_epochs = 100
#num_classes = 10 # No of images to upload in the dataset
num_workers=8
#normalization size in pixels
resizex=112
resizey= 112

embedding_size=768
train_ratio=0.99

#Arcface and Fuzzy Arcface m, tau  and s
margin=0.5
scale_s=64
tauparameter=0.5

#SDG optimizer related for Arcface and FuzzyArcface
learning_rate= 0.01
momentum= 0.9
weight_decay= 5e-4

#Adaface
h_margin=0.33
#Adaptiveface
lambda_margin=0.01

#num classes lfw
num_classes=5749


# GPU AVAILABILITY

# In[8]:


torch.manual_seed(42)  # Set the seed for PyTorch


# In[9]:


if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available")
else:
    print("Single GPU or no GPU available")


# In[10]:


# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[11]:


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[12]:


# If using CUDA (PyTorch)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.


# In[13]:


# Step 2: Set up available GPUs and seed for reproducibility
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

def set_seed(seed=42):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()


# In[14]:


os.getcwd()


# In[15]:


# Step 3: Define image transforms (with normalization)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# In[16]:


### ----------------- Data Extraction -------------------

# Paths to the archives and where to extract them
lfw_tgz_path = os.path.join(base_path, 'lfw.tgz')
lfw_extract_path = os.path.join(base_path, 'extracted', 'lfw/lfw')

# cfp_tar_path = os.path.join(base_path, 'CFP.tar')
# cfp_extract_path = os.path.join(base_path, 'extracted', 'Data/Images')

# jaffedbase_tar_path = os.path.join(base_path, 'jaffedbase.tar')
# jaffedbase_extract_path = os.path.join(base_path, 'extracted', 'jaffedbase/jaffedbase')

# calfw_zip_path = os.path.join(base_path, 'calfw.zip')
# calfw_extract_path = os.path.join(base_path, 'extracted', 'calfw/calfw')

# cplfw_zip_path = os.path.join(base_path, 'cplfw.zip')
# cplfw_extract_path = os.path.join(base_path, 'extracted', 'cplfw/cplfw')


# In[17]:


# Extract tar files (CFP, Jaffedbase, LFW)
for tar_path, extract_path in [#(cfp_tar_path, cfp_extract_path), 
                               #(jaffedbase_tar_path, jaffedbase_extract_path),
                               (lfw_tgz_path, lfw_extract_path)]:
    if not os.path.exists(extract_path):
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_path)



# In[18]:


# # Extract zip files (CALFW, CPLFW)
# for zip_path, extract_path in [(calfw_zip_path, calfw_extract_path), 
#                                (cplfw_zip_path, cplfw_extract_path)]:
#     if not os.path.exists(extract_path):
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_path)



# In[19]:


import os
from PIL import Image
from torch.utils.data import Dataset

class FlatDirectoryImageDataset(Dataset):
    """
    Custom Dataset for loading images from a directory where images
    are stored in subfolders representing class labels.
    """
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Walk through the dataset directory and gather images and labels
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Extract class label from the folder name (person's name)
                    class_name = os.path.basename(os.path.dirname(img_path))
                    
                    # If the class hasn't been seen before, add it to class_to_idx mapping
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)
                    
                    # Assign the numerical label based on class_to_idx mapping
                    label = self.class_to_idx[class_name]
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# In[20]:


### ----------------- Loading Training Dataset (LFW) -------------------

train_dataset = FlatDirectoryImageDataset(lfw_extract_path, transform=transform)


# In[21]:


# from collections import Counter

# # Assuming you're using ImageFolder or a custom dataset with a `targets` attribute
# class_counts = Counter(train_dataset.labels)
# print(f"Class distribution: {class_counts}")


# In[22]:


# DataLoader with num_workers and pin_memory for GPU efficiency
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)



# In[23]:


# test_datasets = {
#     'CPLFW': FlatDirectoryImageDataset(os.path.join(cplfw_extract_path, 'aligned images'), transform),
#     'CALFW': FlatDirectoryImageDataset(os.path.join(calfw_extract_path, 'aligned images'), transform),
#     'JEFF': FlatDirectoryImageDataset(jaffedbase_extract_path, transform),
#     'CFP': ImageFolder(os.path.join(cfp_extract_path, 'Data/Images'), transform),
# }


# In[24]:


# # DataLoader with num_workers and pin_memory for efficient loading
# test_loaders = {name: DataLoader(ds, batch_size=32, shuffle=False, 
#                                  num_workers=num_workers, pin_memory=True) for name, ds in test_datasets.items()}



# In[25]:


# # Iterate over each test loader and print out some samples
# for name, loader in test_loaders.items():
#     print(f"Checking data from {name} dataset:")
    
#     # Get one batch of data
#     data_iter = iter(loader)
    
#     # Try fetching one batch
#     try:
#         images, labels = next(data_iter)  # Get a batch of images and labels
#         print(f"Sample batch from {name}:")
#         print(f"Images shape: {images.shape}")  # Check the shape of the images
#         print(f"Labels: {labels[:5]}")  # Print the first 5 labels
#         print(f"Number of samples in this batch: {len(images)}\n")
#     except StopIteration:
#         print(f"No data found in the {name} loader.\n")


# In[26]:


# Step 6: Define the iResNet100 architecture
class iResNet100(nn.Module):
    def __init__(self, num_classes=num_classes):  # LFW classes
        super(iResNet100, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)



# Define all loss classes

# In[27]:


# class FuzzyArcFaceLoss(nn.Module):
#     def __init__(self, in_features=embedding_size, out_features=num_classes, s=scale_s, m=margin, tau=tauparameter, easy_margin=False):
#         super(FuzzyArcFaceLoss, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)  # Initialize the weights
#         self.tau = tau
#         self.easy_margin = easy_margin

#     def forward(self, input, label):
        
#         #cosine = F.linear(F.normalize(input), F.normalize(self.weight.to(device)))
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight.to(device)).T)

#         # Fuzzy membership based on cosine, values could be -1 to 1
#         fuzzy_membership = cosine
#         # Keep values greater than 0 and greater than tau, and less than or equal to 1
#         mask = torch.logical_and(torch.logical_and(fuzzy_membership > 0, fuzzy_membership >= self.tau), fuzzy_membership <= 1)
#         # Apply mask
#         fuzzy_membership = torch.where(mask, fuzzy_membership, torch.ones_like(fuzzy_membership))
#         # Adjust margin
#         m_adjusted = self.m * fuzzy_membership
        
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
#         # Compute phi with adjusted m
#         cos_m_adjusted = torch.cos(m_adjusted)
#         sin_m_adjusted = torch.sin(m_adjusted)
#         phi = cosine * cos_m_adjusted - sine * sin_m_adjusted

#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > torch.cos(torch.tensor(math.pi) - m_adjusted), phi, cosine - torch.sin(torch.tensor(math.pi) - m_adjusted) * m_adjusted)

#         one_hot = torch.zeros(cosine.size(), device=input.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
        
#         print(f"Output shape: {output.shape}, Labels shape: {labels.shape}")

#         #one_hot.scatter_(1, labels.view(-1, 1), 1)
#         #fuzzy_membership = self.compute_fuzzy_membership(logits, labels)
#         #adjusted_margin = self.margin * fuzzy_membership
#         #logits_with_margin = cosine_sim - one_hot * adjusted_margin
#         #scaled_logits = self.scale * logits_with_margin
#         loss = F.cross_entropy(output, labels)
        
        
#         return loss


# In[28]:


class FuzzyArcFaceLoss(nn.Module):
    def __init__(self, in_features=768, out_features=num_classes, s=scale_s, m=margin, tau=tauparameter, easy_margin=False):
        super(FuzzyArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Note that weight should map from in_features (input embedding size) to out_features (number of classes)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # Shape (num_classes, embedding_size)
        nn.init.xavier_uniform_(self.weight)  # Initialize the weights
        self.tau = tau
        self.easy_margin = easy_margin

    def forward(self, input, label):
        # Normalize input and weight, then compute cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.to(input.device)).T)  # Shape (batch_size, num_classes)

        # Fuzzy membership based on cosine similarity
        fuzzy_membership = cosine
        # Apply the mask
        mask = torch.logical_and(torch.logical_and(fuzzy_membership > 0, fuzzy_membership >= self.tau), fuzzy_membership <= 1)
        fuzzy_membership = torch.where(mask, fuzzy_membership, torch.ones_like(fuzzy_membership))
        
        # Adjust margin
        m_adjusted = self.m * fuzzy_membership
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Compute phi with adjusted margin
        cos_m_adjusted = torch.cos(m_adjusted)
        sin_m_adjusted = torch.sin(m_adjusted)
        phi = cosine * cos_m_adjusted - sine * sin_m_adjusted

        # Apply easy margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > torch.cos(torch.tensor(math.pi) - m_adjusted), phi, cosine - torch.sin(torch.tensor(math.pi) - m_adjusted) * m_adjusted)

        # One-hot encoding for labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Calculate the final output (logits) based on one-hot encoding
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Return the logits for cross-entropy loss outside the function
        return output


# In[29]:


# ArcFace
class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        cosine_sim = F.normalize(logits, dim=1)
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        cosine_sim_with_margin = cosine_sim - one_hot * self.margin
        scaled_logits = self.scale * cosine_sim_with_margin
        loss = F.cross_entropy(scaled_logits, labels)
        return loss



# In[30]:


# AdaptiveFace
class AdaptiveFaceLoss(nn.Module):
    def __init__(self, scale=64.0, base_margin=0.5, lambda_reg=0.001):
        super(AdaptiveFaceLoss, self).__init__()
        self.scale = scale
        self.base_margin = base_margin
        self.lambda_reg = lambda_reg
    
    def compute_adaptive_margin(self, labels, class_counts):
        max_count = class_counts.max().float()
        class_weights = max_count / class_counts.float()
        class_margins = self.base_margin * class_weights[labels]
        return class_margins

    def forward(self, logits, labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_counts = torch.zeros(logits.size(1), device=logits.device)
        #class_counts[unique_labels] = counts
        class_counts[unique_labels] = counts.float()

        
        cosine_sim = F.normalize(logits, dim=1)
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        adaptive_margins = self.compute_adaptive_margin(labels, class_counts)
        margin_per_class = one_hot * adaptive_margins.view(-1, 1)
        logits_with_margin = cosine_sim - margin_per_class
        scaled_logits = self.scale * logits_with_margin
        loss = F.cross_entropy(scaled_logits, labels)
        
        regularization_loss = self.lambda_reg * torch.mean(adaptive_margins)
        total_loss = loss + regularization_loss
        return total_loss



# In[31]:


# VPL (Virtual Prototypical Learning)
class VPLLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5, alpha=0.1):
        super(VPLLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.alpha = alpha

    def compute_virtual_prototypes(self, logits, noise_factor=0.1):
        noise = torch.randn_like(logits) * noise_factor
        virtual_prototypes = logits + noise
        return virtual_prototypes

    def forward(self, logits, labels):
        cosine_sim_real = F.normalize(logits, dim=1)
        one_hot = torch.zeros_like(cosine_sim_real)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        virtual_logits = self.compute_virtual_prototypes(logits)
        cosine_sim_virtual = F.normalize(virtual_logits, dim=1)
        
        real_logits_with_margin = cosine_sim_real - one_hot * self.margin
        virtual_logits_with_margin = cosine_sim_virtual - one_hot * self.margin
        
        scaled_real_logits = self.scale * real_logits_with_margin
        scaled_virtual_logits = self.scale * virtual_logits_with_margin
        
        combined_logits = scaled_real_logits + self.alpha * scaled_virtual_logits
        loss = F.cross_entropy(combined_logits, labels)
        return loss


# In[32]:


# SphereFace2
class SphereFace2Loss(nn.Module):
    def __init__(self, scale=64.0, margin=1.35):
        super(SphereFace2Loss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        cosine_sim = F.normalize(logits, dim=1)
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits_with_margin = cosine_sim * (self.margin * one_hot)
        scaled_logits = self.scale * logits_with_margin
        loss = F.cross_entropy(scaled_logits, labels)
        return loss


# In[33]:


# UniFace
class UniFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5):
        super(UniFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        cosine_sim = F.normalize(logits, dim=1)
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits_with_margin = cosine_sim - one_hot * self.margin
        scaled_logits = self.scale * logits_with_margin
        loss = F.cross_entropy(scaled_logits, labels)
        return loss



# In[34]:


# UniTSFace
class UniTSFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5, temporal_weight=0.5):
        super(UniTSFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.temporal_weight = temporal_weight

    def compute_temporal_logits(self, features, time_step):
        print(features.shape)
        print(time_step.shape)
        logits = F.normalize(features @ time_step.mT, dim=1)
        return logits

    def forward(self, features_t, features_t1, labels):
        logits_t = self.compute_temporal_logits(features_t, features_t)
        logits_t1 = self.compute_temporal_logits(features_t1, features_t1)
        
        cosine_sim_t = F.normalize(logits_t, dim=1)
        cosine_sim_t1 = F.normalize(logits_t1, dim=1)
        
        one_hot = torch.zeros_like(cosine_sim_t)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        logits_t_with_margin = cosine_sim_t - one_hot * self.margin
        logits_t1_with_margin = cosine_sim_t1 - one_hot * self.margin
        
        scaled_logits_t = self.scale * logits_t_with_margin
        scaled_logits_t1 = self.scale * logits_t1_with_margin
        
        loss_t = F.cross_entropy(scaled_logits_t, labels)
        loss_t1 = F.cross_entropy(scaled_logits_t1, labels)
        
        temporal_consistency_loss = F.mse_loss(cosine_sim_t, cosine_sim_t1)
        total_loss = loss_t + loss_t1 + self.temporal_weight * temporal_consistency_loss
        return total_loss



# In[35]:


# Step 8: Define optimizer and training function
def get_optimizer(model, lr=0.01, momentum=0.9):
    return SGD(model.parameters(), lr=lr, momentum=momentum)


# TRAINING

# In[36]:


# Initialize the list for storing model file names
model_filenames = []


# In[37]:


def train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs, loss_name=""):
    model.train()  # Set the model to training mode
    j=0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Debug: Check labels and inputs
            #print(f"Labels: {labels[:5]}")  # Check the first few labels
            #print(f"Inputs shape: {inputs.shape}")
            
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            
           # print(f"Model outputs (first 5): {outputs[:5]}")  # Check model output

            if loss_name=="FuzzyArcFace":
                loss = F.cross_entropy(outputs, labels)
            else:
                loss = criterion(outputs, labels)  # Compute loss
            
            #print(f"Loss value: {loss.item()}")  # Print loss
            
            loss.backward()  # Backpropagation
            
            
            # Check gradients of one layer
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"Gradients of {name}: {param.grad[:5]}")  # Print the first few gradients
            #         break
                    
            
            optimizer.step()  # Update model parameters
            running_loss += loss.item() * inputs.size(0)
            j+=1
            #print(j)
            if j % 10 == 0:  # Print every 10 steps for better monitoring
                print(f"Step {j}, Loss: {loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Save the model with date and loss function in the name
    now = datetime.now().strftime("%Y-%m-%d")
    #now = "2024-09-30"  
    model_name = f"{loss_name}_{now}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
    # Append the saved filename to the list for future loading
    #model_filenames.append(model_name)


# Train all models using different loss functions

# In[38]:


# Step 9: Train all models using different loss functions
loss_functions = {
    "FuzzyArcFace": FuzzyArcFaceLoss(),
    # "ArcFace": ArcFaceLoss(),
    # "AdaptiveFace": AdaptiveFaceLoss(),
    # "VPL": VPLLoss(),
    # "SphereFace2": SphereFace2Loss(),
    # "UniFace": UniFaceLoss(),
   # "UniTSFace": UniTSFaceLoss(),
}


# In[ ]:


# Initialize an empty list to store the trained models
modelss = []

#models = []
now = datetime.now().strftime("%Y-%m-%d")
#now = "2024-09-30"  
# Loop over each loss function
for loss_name, loss_fn in loss_functions.items():
    model_file = os.path.join(model_save_path, f'{loss_name}_{now}.pth')

    # Check if the model file exists
    if os.path.exists(model_file):
        print(f"Model for {loss_name} already exists. Skipping training...")
        # Load the model from the file
        # model = iResNet100()
        # model.load_state_dict(torch.load(model_file))
        # model = model.to(device)
        # modelss.append(model)
        model_name = f"{loss_name}_{now}.pth"
        model_filenames.append(model_name)
    else:
        print(f"Training with {loss_name}...")
        model = iResNet100()  # New model for each loss function
        model = nn.DataParallel(model) if num_gpus > 1 else model  # DataParallel if multiple GPUs
        model = model.to(device)
        optimizer = get_optimizer(model)
        train_model(model, loss_fn, optimizer, train_loader, num_epochs=num_epochs, loss_name=loss_name)
        model_name = f"{loss_name}_{now}.pth"
        model_filenames.append(model_name)
        modelss.append(model)



# In[ ]:




