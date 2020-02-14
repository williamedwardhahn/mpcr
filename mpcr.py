from __future__ import print_function, division

name='google_images_download'

import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

install(name)

from google_images_download import google_images_download
from google.colab import files
from google.colab import drive
import shutil
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch import tensor 
from torch.autograd import Variable
from torch import randn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torchsummary import summary
from IPython.core.display import display, HTML

import tensorflow as tf
import scipy.io
from scipy.cluster.vq import whiten
from sklearn.preprocessing import scale
import h5py
from skimage import io
import tflearn
import tflearn.datasets.mnist as mnist
import tensorflow as tf
from skimage import io as io



from scipy import signal
from skimage.util import view_as_blocks, view_as_windows
from skimage.transform import rescale, resize

from scipy.misc import bytescale
import plotly.plotly as py
import plotly.graph_objs as go
from matplotlib import pylab
from numpy import arange,array,ones
from scipy import stats
import numpy as np








from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17

import warnings
warnings.filterwarnings("ignore")



plt.ion()   # interactive mode


data_dir = '/content/Data/'



def clean_folders(s1,s2):
  
  
  folder = '/content/Data/Train/'+s1
  
  for filename in os.listdir(folder):
  
    f = os.path.join(folder,filename)

    img = cv2.imread(f)

    if img is None:

      os.remove(f)
  
  folder = '/content/Data/Train/'+s2
  
  for filename in os.listdir(folder):
  
    f = os.path.join(folder,filename)

    img = cv2.imread(f)

    if img is None:

      os.remove(f)
    
heads = 'https://random-ize.com/coin-flip/us-quarter/us-quarter-front.jpg'

tails = 'https://random-ize.com/coin-flip/us-quarter/us-quarter-back.jpg'    
    
def show_img(img):
  
  display(HTML('<img src=' + img + '>'))
  

    
def split_train_test(source,dest):

  dirs = os.listdir(source)

  for d in dirs:
    
    print(d)

    os.mkdir(dest+'/'+d)

    files = os.listdir(source+'/'+d)

    for f in files[:int(len(files)/5)]: #1/5 = %20 of the data is moved to testing

      print(f)

      shutil.move(source+'/'+d+'/'+f, dest+'/'+d+'/'+f)    
    

def get_images_web(s1,s2):
  
  os.mkdir('/content/Data')
  os.chdir('/content/Data')
  os.mkdir('Train')
  os.mkdir('Test')
  
  arguments = {"keywords":s1 +","+ s2,"limit":100,"print_urls":False,"output_directory":'/content/Data/Train',"safe_search":True}   #creating list of arguments

  response = google_images_download.googleimagesdownload()
  
  paths = response.download(arguments)   #passing the arguments to the function
    
  clean_folders(s1,s2)

  source = '/content/Data/Train'
  dest = '/content/Data/Test'

  split_train_test(source,dest)  

  dataloaders,dataset_sizes,class_names,device = load_data(data_dir)  
  
  return dataloaders,class_names,dataset_sizes,device  

def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()        
        

def load_data(data_dir):
  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
      'Train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'Test': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }


  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Test']}
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['Train', 'Test']}
  dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}
  class_names = image_datasets['Train'].classes
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  return dataloaders,dataset_sizes,class_names,device        
        
 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
    fig, ax = plt.subplots()
    im = ax.imshow(inp, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(18, 10)
    
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
    
    
    
    
    
def train_model(dataloaders,class_names,dataset_sizes,device, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in Train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model    
    
def train_network(dataloaders,class_names,dataset_sizes,device):
  
  model_ft = models.resnet18(pretrained=True)

  model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)

  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
  
  model_ft = train_model(dataloaders,class_names,dataset_sizes,device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
  
  return model_ft


def show_output(model, dataloaders, class_names, device, num_images=16):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['Test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
        
def show_batch(dataloaders,class_names):

  inputs, classes = next(iter(dataloaders['Train']))# Get a batch of training data
  imshow(torchvision.utils.make_grid(inputs), title=[class_names[x] for x in classes])
    
    
def save_model(model,s1,s2):
  
  torch.save(model.state_dict(), '/content/Data/'+s1+" "+s2 + ".mpcr")
    
    
def load_model(model_name):
  
  model = models.resnet18(pretrained=True)

  model.fc = nn.Linear(model.fc.in_features, 2)

  model = model.to(device=torch.device('cuda:0'))
  
  model.load_state_dict(torch.load(model_name))
  
  model.eval()
  
  return model



def montage(X):    
    m, n, count = np.shape(X)    
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count: 
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = X[:, :, image_id]
            image_id += 1
    return M

def montage2(X):    
    m, n, count = np.shape(X)    
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count: 
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = np.transpose(X[:, :, image_id])
            image_id += 1
    return M

def tensor_montage_plot(data):
    P=np.dstack((montage2(np.transpose(data[:,:,:,0])),montage2(np.transpose(data[:,:,:,1])),montage2(np.transpose(data[:,:,:,2]))))
    return P


def mat2ten(X):
    zs=[int(np.sqrt(X.shape[0])),int(np.sqrt(X.shape[0])),X.shape[1]]
    Z=np.zeros(zs)
    for i in range(X.shape[1]):
        Z[:,:,i]=np.reshape(X[:,i],[zs[0],zs[1]])
    return Z
