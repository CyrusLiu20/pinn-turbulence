# Third party packages
import torch
from torch import nn
from torch import autograd
from torchsummary import summary
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyDOE import lhs         #Latin Hypercube Sampling

#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Physics informed neural network for turbulence modelling")
print(f"{device} : currently in use")

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 