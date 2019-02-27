from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict
from csbdeep.io import save_training_data
from csbdeep.data import RawData, create_patches
from csbdeep.data.transform import anisotropic_distortions
import os



#set the raw directory of your image set
#eg of usage: 
#full_img_dir == C:/Users/e0031794/Desktop/animal_pic
#a subfolder in the img_dir == retina
#RETURNS a raw data object
def set_raw_data_dir(full_img_dir, img_target_sub_folder):

    base_path = full_img_dir
    source_dir = [img_target_sub_folder]
    target_dir = img_target_sub_folder

    raw_data = RawData.from_folder(base_path, source_dir, target_dir, 'ZCYX')
    return raw_data

#takes in a parameter for subsample factor
#returns an anisotropic transform object
def set_anisotropic_transform(subsample_factor):

    anisotropic_transform = anisotropic_distortions(
        subsample = subsample_factor,
        psf = np.ones((3,3))/9,
        psf_axes = 'YX',
    )
    return anisotropic_transform

#takes in anisotropic_transform object (returned from set_anisotropic_transform) & raw data object (returned from set_raw_data_dir), and a desired storage directory name
#returns a preprocessed set of training data, in which the z-axis of the image is removed.
#hence, this function has 2 steps: 1. to create anisotropic distortions with a given microscopic image, 2. to remove z-axis from distorted image and store it in a desired dir
#3. the storage dir will be in the same folder as this python programme and its name == desired storage dir name

def generate_anisotropic_training_data(raw_data_obj, anisotropic_transform_obj, storage_dir):

    #get our input & distorted trng img set and their corresponding labels set, and the img set axes
    input_trng_img_set, output_trng_label_set, img_XY_axes = create_patches(
        raw_data = raw_data_obj,
        patch_size = (1,2,128, 128),
        n_patches_per_image = 512,
        transforms = [anisotropic_transform_obj]
    )

    #z-axis removal
    z = axes_dict(img_XY_axes)['Z']
    input_trng_img_set = np.take(input_trng_img_set, 0, axis=z)
    output_trng_label_set = np.take(output_trng_label_set, 0, axis=z)
    img_XY_axes = img_XY_axes.replace('Z', '')

    #storing training data
    #get current working dir of this python programme
    cwd = os.getcwd()

    #create storage folder
    full_storage_dir = os.path.join(cwd, storage_dir)
    os.mkdir(full_storage_dir)

    training_data_name = os.path.join(full_storage_dir, 'training_data.npz')
    save_training_data(training_data_name, input_trng_img_set, output_trng_label_set, img_XY_axes)

    return training_data_name







    
