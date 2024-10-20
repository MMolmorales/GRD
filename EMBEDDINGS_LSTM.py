# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:10:56 2024

@author: mirko
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import seaborn as sns 
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


dataset = pd.read_csv("dataset_elpino.csv", sep = ";" , on_bad_lines='skip') 

#%% Modificación del dataset

dataset.rename(columns={'Edad en años': 'Edad'}, inplace=True)
dataset.rename(columns={'Sexo (Desc)': 'Sexo'}, inplace=True)
dataset['GRD'] = dataset['GRD'].str.slice(0, 5)
Y = dataset['GRD'].iloc[:]

    
    
    
columnas = dataset.columns
for i in range(len(columnas)-3):
    dataset[columnas[i]] = dataset[columnas[i]].str.slice(0, 4)



GRD_top100 = []
Conteo_otros = [['99999', 0]]
GRD_otros = []
GRD_top20 = []
valores_unicos, conteos = np.unique(Y, return_counts=True)
for i in range(len(valores_unicos)):
    if conteos[i] >= 20:
        GRD_top20.append(valores_unicos[i])
        if conteos[i] >= 100:
            GRD_top100.append(valores_unicos[i])
    else:
        Conteo_otros[0][1]+= conteos[i] 
        GRD_otros.append(valores_unicos[i])

GRD_comunes = ['14610', '14612' , '14613'] 

dataset_filtrado = dataset
#dataset_filtrado['GRD'] = dataset_filtrado['GRD'].replace(GRD_otros, '99999') # Todos los registros
dataset_filtrado = dataset[dataset['GRD'].isin(GRD_top100)] # Solo registros más frecuentes


Secuencial = dataset_filtrado.drop(columns = ['Edad','Sexo'])
nfilas = len(Secuencial)
Y = Secuencial['GRD'].iloc[:]

columnas_S = Secuencial.columns





#%% TOKENIZAR


import string 

translator = str.maketrans(dict.fromkeys(string.punctuation)) # eliminar signos de puntuación


Token_s = Secuencial.iloc[:nfilas]


# Iterar sobre cada columna
for col in Token_s.columns:
 Token_s[col] = Token_s[col].apply(lambda x: word_tokenize(str(x).translate(translator)))

Token_s['combined'] = Token_s.apply(lambda row: ' '.join(sum(row.tolist(), [])), axis=1) 
Token_s['combined'] = Token_s['combined'].apply(lambda x : word_tokenize(str(x)))

 
#%% Creación Matriz Caracteristicas

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical



X = Token_s['combined']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
max_length = max(len(seq) for seq in sequences)  # Longitud máxima de las secuencias
X = pad_sequences(sequences, maxlen=max_length, padding='post')

#%% OHE Y

class_mapping = {value: idx for idx, value in enumerate(np.unique(Y))}
Y_indices = np.array([class_mapping[value] for value in Y])
Y_categorical = to_categorical(Y_indices)
nclass = Y_categorical.shape[1]

#%% Dividir en train val

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y_categorical, test_size=0.2, stratify=Y_categorical, random_state=123)



#%% EMBEDDING + LSTM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
vocabulario = len(tokenizer.word_index) + 1  # Tamaño del vocabulario
embedding_dim = 200  # Dimensión del embedding # Si es para 3 GRD hay que cambiar el valor, porque 

model = Sequential()
model.add(Embedding(input_dim=vocabulario, output_dim=embedding_dim, input_length=max_length)) 
model.add(LSTM(200, return_sequences=True, input_shape=( x_train.shape[1], embedding_dim )))  
model.add(BatchNormalization())
model.add(Dropout(0.3))  
model.add(LSTM(150, return_sequences=True, input_shape=( x_train.shape[1], embedding_dim )))  
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(30, input_shape = (x_train.shape[1], embedding_dim)))  
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))  
model.add(Dropout(0.2))
model.add(Dense(nclass, activation='softmax'))  # Salida con nclass clases


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#%%


model.summary()


#%%

history = model.fit(x_train,y_train,validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr]) 


#%%
#model.save("modelo_3.keras")      # Para guardar el modelo usando GRD_comunes
#model.save("modelo_top100.keras") # Para guardar el modelo con GRD_top100
#model.save("modelo_ALL.keras")    # Para guardar el modelo usando todos los datos.


#%% Predecir con el modelo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

umbral=0.5
y_pred_probs=model.predict(x_val)
y_test = y_val
y_pred=y_pred_probs[:]>=umbral

y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

#%% Matriz de confusión
# Usar solo para GRD menores a 5, con más clases no se va a entender. 

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
Size = 30
# Visualizar la matriz de confusión
plt.figure(figsize=(20,15))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": Size})
ax.set_xlabel('Predicted labels',  fontsize = Size)
ax.set_ylabel('True labels', fontsize = Size)
ax.set_xticklabels(GRD_comunes, ha='right', fontsize=Size)
ax.set_yticklabels(GRD_comunes, ha='right', fontsize=Size)
ax.set_title(f'Confusion Matrix + LSTM {nclass} GRD',  fontsize = Size)
plt. show()


#%% Curvas de precisión y costo.                     

Size = 30

plt.figure(figsize=(15,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Costo', fontsize = Size)
plt.xlabel('Época', fontsize = Size)
plt.xticks( fontsize = Size/3*2)
plt.yticks(fontsize = Size*2/3)
plt.legend(['Entrenamiento', 'Validación'], loc='upper right', fontsize = Size)
plt.title(f'Red Embedding + LSTM {nclass} GRD', fontsize = Size)
plt.grid(True)
plt.show()

# Graficar la precisión
plt.figure(figsize=(15,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Precisión', fontsize = Size)
plt.xlabel('Época', fontsize = Size)
plt.xticks( fontsize = Size/3*2)
plt.yticks(fontsize = Size*2/3)
plt.legend(['Entrenamiento', 'Validación'], loc='lower right', fontsize = Size)
plt.title(f'Red Embedding + LSTM {nclass} GRD', fontsize = Size)
plt.grid(True)
plt.show()


#%%




