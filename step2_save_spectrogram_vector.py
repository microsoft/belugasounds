#
# save_spectrogram_vector.py
#
# Reformat spectraogram data into training-ready .csv files
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import os
from fnmatch import fnmatch
import cv2
import numpy as np
import random


#%% Path configuration

current_dir = "./Whale_Acoustics/"
model_dir = current_dir + "Model/"
data_dir = current_dir + "Data/"
spectrogram_dir = data_dir + "Extracted_Spectrogram/"
output_spectrogram_vector_dir = "Output_Spectrogram_Vector/"

if not os.path.exists(data_dir + output_spectrogram_vector_dir):
    os.makedirs(data_dir + output_spectrogram_vector_dir)


#%% Load spectrograms
    
spectrograms_B = []
spectrograms_F = []
spectrograms_N = []
filenames_B = []
filenames_F = []
filenames_N = []

ncol, nrow = 300, 300
index = 0

for path, subdirs, files in os.walk(spectrogram_dir):
    for filename in files:
        if(index % 1000 == 0):
            print(index)
        try:
            if fnmatch(filename, '*_B.png'):
                img = cv2.imread(path + '/' + filename)
                img = cv2.resize(img, (ncol, nrow))
                spectrograms_B.append(img)
                filenames_B.append(path + '/' + filename)
            elif fnmatch(filename, '*_F.png'):
                img = cv2.imread(path + '/' + filename)
                img = cv2.resize(img, (ncol, nrow))
                spectrograms_F.append(img)
                filenames_F.append(path + '/' + filename)
            if fnmatch(filename, '*_N.png'):
                img = cv2.imread(path + '/' + filename)
                img = cv2.resize(img, (ncol, nrow))
                spectrograms_N.append(img)
                filenames_N.append(path + '/' + filename)
        except:
            pass
        index += 1

print(len(spectrograms_B))
print(len(spectrograms_F))
print(len(spectrograms_N))

spectrograms_B = np.asarray(spectrograms_B)
spectrograms_F = np.asarray(spectrograms_F)
spectrograms_N = np.asarray(spectrograms_N)

np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_B_300_300", spectrograms_B)
np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_F_300_300", spectrograms_F)
np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_N_300_300", spectrograms_N)

with open(data_dir + output_spectrogram_vector_dir + "filenames_B.csv",'w') as f:
    for filename in filenames_B:
        f.write(filename)
        f.write('\n')

with open(data_dir + output_spectrogram_vector_dir + "filenames_F.csv",'w') as f:
    for filename in filenames_F:
        f.write(filename)
        f.write('\n')

with open(data_dir + output_spectrogram_vector_dir + "filenames_N.csv",'w') as f:
    for filename in filenames_N:
        f.write(filename)
        f.write('\n')


#%% Generate a random sample of spectrograms for training
        
random.seed(40)
spectrograms_B_sample_index = sorted(random.sample(range(len(filenames_B)), 50000))
spectrograms_F_sample_index = sorted(random.sample(range(len(filenames_F)), 100000))
spectrograms_N_sample_index = sorted(random.sample(range(len(filenames_N)), 20000))

spectrograms_B_sample = spectrograms_B[spectrograms_B_sample_index]
spectrograms_F_sample = spectrograms_F[spectrograms_F_sample_index]
spectrograms_N_sample = spectrograms_N[spectrograms_N_sample_index]
        
filenames_B_sample = np.asarray(filenames_B)[spectrograms_B_sample_index].tolist()
filenames_F_sample = np.asarray(filenames_F)[spectrograms_F_sample_index].tolist()
filenames_N_sample = np.asarray(filenames_N)[spectrograms_N_sample_index].tolist()

np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_B_sample_300_300", spectrograms_B_sample)
np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_F_sample_300_300", spectrograms_F_sample)
np.save(data_dir + output_spectrogram_vector_dir + "spectrograms_N_sample_300_300", spectrograms_N_sample)

with open(data_dir + output_spectrogram_vector_dir + "filenames_B_sample.csv",'w') as f:
    for filename in filenames_B_sample:
        f.write(filename)
        f.write('\n')

with open(data_dir + output_spectrogram_vector_dir + "filenames_F_sample.csv",'w') as f:
    for filename in filenames_F_sample:
        f.write(filename)
        f.write('\n')

with open(data_dir + output_spectrogram_vector_dir + "filenames_N_sample.csv",'w') as f:
    for filename in filenames_N_sample:
        f.write(filename)
        f.write('\n')
