from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
#import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, IsotropicCARE
from generate_trng_data import *
import argparse

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

    parser = argparse.ArgumentParser(description='Generate trng data and train model w deep learning')
    parser.add_argument('--d', type=str, help='Full path of image folder')
    parser.add_argument('--t', type=str, help='Target subfolder within image folder')
    parser.add_argument('--sf', type=float, default=10.2, help='subsample_factor for anisotropic transform')
    parser.add_argument('--str', type=str, help='storage folder for transformed training data')

    args = parser.parse_args()
    print('setting raw data directory ...')
    raw_data = set_raw_data_dir(args.d, args.t)
    print('setting anisotropic transform ...')
    anisotropic_transform = set_anisotropic_transform(args.sf)
    print('generating anisotropic training data...')
    full_trng_data_path = generate_anisotropic_training_data(raw_data, anisotropic_transform, args.str)
    print('training network....')
    trng_history = train(full_trng_data_path, 10, 30)
    print('done!')


