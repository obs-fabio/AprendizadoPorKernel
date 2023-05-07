import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from keras.layers.core import Dense, Dropout, Activation
import keras.callbacks as callbacks
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from Models import auc_score, sensitivity_score, specificity_score, sp_index, acc_score, get_confusion_matrix

weigth_std = 0.1
learning_rate = 0.05
epochs=200
batch_size=1
patience = 100
verbose=True

df = pd.read_csv('200-iara-relevant.csv', sep=',')

data = df[['Hora','PMA','Speed','SPL','SNR','Calado','Direct','Indirect','Detections','Alone']]
output_label = 'Ideal'
target = df[[output_label]] # 'Visible','Ideal'


n_inputs = data.shape[1]
n_outputs = 1#target.shape[1]
n_samples = data.shape[0]
batch_size = int(batch_size*n_samples)

random.seed(42)
ids = np.arange(1, n_samples)
ids = np.random.permutation(ids)

trn_idx = ids[:round(n_samples*0.8)]
val_idx = ids[round(n_samples*0.8):round(n_samples*0.9)]
test_idx = ids[round(n_samples*0.9):]

trn_target = ((target[output_label].values)[trn_idx]+1)/2
val_target = ((target[output_label].values)[val_idx]+1)/2

model = Sequential()
model.add(Dense(n_outputs, input_dim=n_inputs, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit(data.iloc[trn_idx,:],
        trn_target,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(data.iloc[val_idx,:],
                        val_target))

prediction = model.predict(data.iloc[test_idx,:])

y_prediction = (prediction > 0).astype(int)
y_target = ((target[output_label].values)[test_idx]+1)/2.

print("auc_score: ", auc_score(y_target, y_prediction))
print("sensitivity_score: ", sensitivity_score(y_target, y_prediction))
print("specificity_score: ", specificity_score(y_target, y_prediction))
print("sp_index: ", sp_index(y_target, y_prediction))
print("acc_score: ", acc_score(y_target, y_prediction))
print("get_confusion_matrix: ", get_confusion_matrix(y_target, y_prediction))

# model = tf.keras.Sequential()

# model.add(tf.keras.Input(shape=(n_inputs,)))

# output_layer = layers.Dense(units=n_outputs,
#                             activation='tanh',
#                             kernel_initializer=initializers.RandomNormal(stddev=weigth_std),
#                             bias_initializer=initializers.RandomNormal(stddev=weigth_std))
# model.add(output_layer)

# opt = keras.optimizers.Adam(learning_rate=learning_rate,
#                             beta_1=0.9,beta_2=0.999,
#                             epsilon=1e-07,amsgrad=False,
#                             name="Adam",)

# loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)


# cat_acc_metric = keras.metrics.CategoricalAccuracy(name="cat_acc", dtype=None)
# acc_metric = keras.metrics.Accuracy(name="accuracy",dtype=None)
# mse_metric = keras.metrics.MeanSquaredError(name="mse", dtype=None)
# rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)

# model.compile(loss=loss, 
#                 optimizer=opt,
#                 metrics=[cat_acc_metric,
#                         acc_metric,
#                         mse_metric,
#                         rmse_metric])

# earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
#                                         patience=patience,verbose=verbose,
#                                         mode='auto')

# trn_desc = model.fit(data[trn_idx,:], target[trn_idx],
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         callbacks=[earlyStopping],
#                         verbose=verbose,
#                         validation_data=(data[val_idx,:],
#                                         target[val_idx]),
#                     )


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./data/plots/erro.png')