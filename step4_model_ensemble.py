#
# model_ensemble.py
#
# Train the ensemble model
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras imports
from keras.models import model_from_json
from scipy import optimize
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


#%% Path configuration

current_dir = "./Whale_Acoustics/"
model_dir = current_dir + "Model/"
data_dir = current_dir + "Data/"
output_dir = current_dir + 'Output/'
spectrogram_dir = data_dir + "Extracted_Spectrogram/"
output_spectrogram_vector_dir = "Output_Spectrogram_Vector/"


#%% Step 1: train/validation/test split

ncol, nrow = 300, 300

spectrograms_B_sample = np.load(data_dir + output_spectrogram_vector_dir + "spectrograms_B_sample_300_300.npy")
spectrograms_F_sample = np.load(data_dir + output_spectrogram_vector_dir + "spectrograms_F_sample_300_300.npy")
spectrograms_N_sample = np.load(data_dir + output_spectrogram_vector_dir + "spectrograms_N_sample_300_300.npy")

filenames_B_sample = []
with open(data_dir + output_spectrogram_vector_dir + "filenames_B_sample.csv", newline='') as f:
    for row in csv.reader(f):
        filenames_B_sample.append(row[0])

filenames_F_sample = []
with open(data_dir + output_spectrogram_vector_dir + "filenames_F_sample.csv", newline='') as f:
    for row in csv.reader(f):
        filenames_F_sample.append(row[0])

filenames_N_sample = []
with open(data_dir + output_spectrogram_vector_dir + "filenames_N_sample.csv", newline='') as f:
    for row in csv.reader(f):
        filenames_N_sample.append(row[0])

spectrograms_B_train_validation, spectrograms_B_test, filenames_B_train_validation, filenames_B_test = train_test_split(spectrograms_B_sample, filenames_B_sample, test_size = 0.3, random_state = 1)
spectrograms_F_train_validation, spectrograms_F_test, filenames_F_train_validation, filenames_F_test = train_test_split(spectrograms_F_sample, filenames_F_sample, test_size = 0.3, random_state = 1)
spectrograms_N_train_validation, spectrograms_N_test, filenames_N_train_validation, filenames_N_test = train_test_split(spectrograms_N_sample, filenames_N_sample, test_size = 0.3, random_state = 1)

spectrograms_train_validation = np.concatenate((spectrograms_B_train_validation, spectrograms_F_train_validation, spectrograms_N_train_validation), axis=0)
labels_train_validation = np.array([1] * len(spectrograms_B_train_validation) + [0] * len(spectrograms_F_train_validation) + [0] * len(spectrograms_N_train_validation))

X_train, X_validation, y_train, y_validation = train_test_split(spectrograms_train_validation, labels_train_validation, test_size = 0.3, random_state = 1)


X_train = X_train / 255.0
X_validation = X_validation / 255.0

print(X_train.shape)   
print(X_validation.shape)   
print(spectrograms_B_test.shape)   
print(spectrograms_F_test.shape)
print(spectrograms_N_test.shape)


#%% Step 2: load models

with open(model_dir + 'cnn_architecture_all_data.json', 'r') as f:
    model_cnn = model_from_json(f.read())
model_cnn.load_weights(model_dir + 'cnn_weights_all_data.h5')

with open(model_dir + 'vgg16_architecture_all_data.json', 'r') as f:
    model_vgg16 = model_from_json(f.read())
model_vgg16.load_weights(model_dir + 'vgg16_weights_all_data.h5')

with open(model_dir + 'ResNet50_architecture_all_data.json', 'r') as f:
    model_ResNet50 = model_from_json(f.read())
model_ResNet50.load_weights(model_dir + 'ResNet50_weights_all_data.h5')


with open(model_dir + 'DenseNet121_architecture_all_data.json', 'r') as f:
    model_DenseNet121 = model_from_json(f.read())
model_DenseNet121.load_weights(model_dir + 'DenseNet121_weights_all_data.h5')


#%% Step 3: predict on the validation and test sets

validation_predict_cnn = model_cnn.predict(X_validation) 
validation_predict_cnn = [x for sublist in validation_predict_cnn.tolist() for x in sublist]

validation_predict_vgg16 = model_vgg16.predict(X_validation)
validation_predict_vgg16 = [x for sublist in validation_predict_vgg16.tolist() for x in sublist]

validation_predict_ResNet50 = model_ResNet50.predict(X_validation)
validation_predict_ResNet50 = [x for sublist in validation_predict_ResNet50.tolist() for x in sublist]

validation_predict_DenseNet121 = model_DenseNet121.predict(X_validation)
validation_predict_DenseNet121 = [x for sublist in validation_predict_DenseNet121.tolist() for x in sublist]

validation_predict = [validation_predict_cnn, validation_predict_vgg16, validation_predict_ResNet50, validation_predict_DenseNet121]

# Optimize weights for each model
def f(weights):
    validation_predict_ensemble = np.average(validation_predict, axis=0, weights=weights)
    validation_predict_ensemble_class = [int(validation_predict_ensemble[i] > 0.5) for i in range(len(validation_predict_ensemble))]
    return validation_predict_ensemble_class

def loss_function(weights):
    validation_predict_ensemble_class = f(weights)
    n_lost = [prediction != label for prediction, label in zip(validation_predict_ensemble_class, y_validation)]
    return np.sum(n_lost) / len(y_validation)

model_cnt = 4 # the number of models for ensembling

opt_weights = optimize.minimize(loss_function,
                                [1/ model_cnt] * model_cnt,
                                constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'Nelder-Mead', #'SLSQP',
                                bounds=[(0.0, 1.0)] * model_cnt,
                                options = {'ftol':1e-3},
                            )['x']

print('Optimum weights = ', opt_weights, 'with loss', loss_function(opt_weights))

# Save the optimal weights of each individual model
opt_weights[-1] = 1.0 - sum(opt_weights[:-1])  ## to force the total weights sums up to 1 
pd.DataFrame(opt_weights).to_excel(output_dir + 'opt_weights.xlsx', header=False, index=False)

del X_train
del X_validation
del spectrograms_train_validation
del spectrograms_B_train_validation
del spectrograms_F_train_validation
del spectrograms_N_train_validation
del spectrograms_B_sample
del spectrograms_F_sample
del spectrograms_N_sample

############### CNN: prediction ##############
spectrograms_B_test_predict_cnn = model_cnn.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_cnn = [x for sublist in spectrograms_B_test_predict_cnn.tolist() for x in sublist]

spectrograms_F_test_predict_cnn = model_cnn.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_cnn = [x for sublist in spectrograms_F_test_predict_cnn.tolist() for x in sublist]

spectrograms_N_test_predict_cnn = model_cnn.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_cnn = [x for sublist in spectrograms_N_test_predict_cnn.tolist() for x in sublist]

############### VGG16: prediction ##############
spectrograms_B_test_predict_vgg16 = model_vgg16.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_vgg16 = [x for sublist in spectrograms_B_test_predict_vgg16.tolist() for x in sublist]

spectrograms_F_test_predict_vgg16 = model_vgg16.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_vgg16 = [x for sublist in spectrograms_F_test_predict_vgg16.tolist() for x in sublist]

spectrograms_N_test_predict_vgg16 = model_vgg16.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_vgg16 = [x for sublist in spectrograms_N_test_predict_vgg16.tolist() for x in sublist]

############### ResNet50: prediction ##############
spectrograms_B_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_ResNet50 = [x for sublist in spectrograms_B_test_predict_ResNet50.tolist() for x in sublist]

spectrograms_F_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_ResNet50 = [x for sublist in spectrograms_F_test_predict_ResNet50.tolist() for x in sublist]

spectrograms_N_test_predict_ResNet50 = model_ResNet50.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_ResNet50 = [x for sublist in spectrograms_N_test_predict_ResNet50.tolist() for x in sublist]

############### DenseNet121: prediction ##############
spectrograms_B_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_B_test / 255.0)
spectrograms_B_test_predict_DenseNet121 = [x for sublist in spectrograms_B_test_predict_DenseNet121.tolist() for x in sublist]

spectrograms_F_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_F_test / 255.0)
spectrograms_F_test_predict_DenseNet121 = [x for sublist in spectrograms_F_test_predict_DenseNet121.tolist() for x in sublist]

spectrograms_N_test_predict_DenseNet121 = model_DenseNet121.predict(spectrograms_N_test / 255.0)
spectrograms_N_test_predict_DenseNet121 = [x for sublist in spectrograms_N_test_predict_DenseNet121.tolist() for x in sublist]

############### emsemble: prediction ##############
spectrograms_B_test_predict = [spectrograms_B_test_predict_cnn, spectrograms_B_test_predict_vgg16, spectrograms_B_test_predict_ResNet50, spectrograms_B_test_predict_DenseNet121]
spectrograms_B_test_predict_ensemble = np.average(spectrograms_B_test_predict, axis=0, weights = opt_weights)
spectrograms_B_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v < 0.5]
plt.hist(spectrograms_B_test_predict_ensemble)

spectrograms_F_test_predict = [spectrograms_F_test_predict_cnn, spectrograms_F_test_predict_vgg16, spectrograms_F_test_predict_ResNet50, spectrograms_F_test_predict_DenseNet121]
spectrograms_F_test_predict_ensemble = np.average(spectrograms_F_test_predict, axis=0, weights = opt_weights)
spectrograms_F_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v > 0.5]
plt.hist(spectrograms_F_test_predict_ensemble)

spectrograms_N_test_predict = [spectrograms_N_test_predict_cnn, spectrograms_N_test_predict_vgg16, spectrograms_N_test_predict_ResNet50, spectrograms_N_test_predict_DenseNet121]
spectrograms_N_test_predict_ensemble = np.average(spectrograms_N_test_predict, axis=0, weights = opt_weights)
spectrograms_N_test_predict_ensemble_wrong_predictions = [i for i,v in enumerate(spectrograms_N_test_predict_ensemble) if v > 0.5]
plt.hist(spectrograms_N_test_predict_ensemble)

print(1 - len(spectrograms_B_test_predict_ensemble_wrong_predictions) / len(spectrograms_B_test))  ## 92.26%
print(1 - len(spectrograms_F_test_predict_ensemble_wrong_predictions) / len(spectrograms_F_test))  ## 98.36%
print(1 - len(spectrograms_N_test_predict_ensemble_wrong_predictions) / len(spectrograms_N_test))  ## 99.9%

print(len(spectrograms_B_test_predict_ensemble_wrong_predictions))
print(len(spectrograms_F_test_predict_ensemble_wrong_predictions))
print(len(spectrograms_N_test_predict_ensemble_wrong_predictions))

plt.hist(spectrograms_B_test_predict_ensemble)
plt.xlabel('Predicted Probability')
plt.ylabel('Count of Spectrograms')

plt.hist(spectrograms_F_test_predict_ensemble)
plt.xlabel('Predicted Probability')
plt.ylabel('Count of Spectrograms')

plt.hist(spectrograms_B_test_predict_ensemble[spectrograms_B_test_predict_ensemble_wrong_predictions])
plt.hist(spectrograms_F_test_predict_ensemble[spectrograms_F_test_predict_ensemble_wrong_predictions])
plt.hist(spectrograms_N_test_predict_ensemble[spectrograms_N_test_predict_ensemble_wrong_predictions])

tp = len([i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v >= 0.5])
fn = len([i for i,v in enumerate(spectrograms_B_test_predict_ensemble) if v < 0.5])
tn = len([i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v < 0.5])
fp = len([i for i,v in enumerate(spectrograms_F_test_predict_ensemble) if v >= 0.5])
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + fp + tn + fn)

y_true = [1] * len(spectrograms_B_test_predict_ensemble) + [0] * len(spectrograms_F_test_predict_ensemble)
y_scores = spectrograms_B_test_predict_ensemble.tolist() + spectrograms_F_test_predict_ensemble.tolist()

# Calculate ROC and AUC
AUC = roc_auc_score(y_true, y_scores) 
print('AUC: %.4f' % AUC)

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision_recall_auc = auc(recall, precision)
print('Precesion Recall AUC: %.4f' % precision_recall_auc)

# Plot precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

