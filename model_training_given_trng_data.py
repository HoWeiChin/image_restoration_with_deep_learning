from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, IsotropicCARE
from generate_trng_data import *

def config_model(axes, n_channel_in, n_channel_out, epoch):
    config = Config(axes, n_channel_in, n_channel_out, epoch)
    return config

#accepts full storage dir as an input, validation_percentage as an integer value, for eg 10%, 20% or 30%
def train(full_storage_dir, validation_percentage, epoch):

    val_value = round(validation_percentage/100, 1)
    (X, Y), (X_val, Y_val), axes = load_training_data(full_storage_dir, val_value, verbose=True)

    val_data = (X_val, Y_val)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    config = config_model(axes, n_channel_in, n_channel_out, epoch)
    model = IsotropicCARE(config, 'my_model', basedir='models')

    trng_history = model.train(X, Y, validation_data= val_data)

    return trng_history




if __name__ == '__main__':
    #change these when needed
    full_img_dir = 'C:/Users/e0031794/Desktop/animal_pic'
    target_dir = 'retina'
    subsample_factor = 10.2
    storage_name = 'training_set'

    #running data generation pipeline
    raw_data = set_raw_data_dir(full_img_dir, target_dir)
    anisotropic_transform = set_anisotropic_transform(subsample_factor)
    full_trng_data_path = generate_anisotropic_training_data(raw_data, anisotropic_transform, storage_name)

    trng_history = train(full_trng_data_path, 10, 30)