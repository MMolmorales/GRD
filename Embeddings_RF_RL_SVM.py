# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 20:56:26 2024

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
dataset_filtrado = dataset[dataset['GRD'].isin(GRD_comunes)] # Solo registros más frecuentes
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
x_train, x_val, y_train, y_val = train_test_split(X, Y_indices, test_size=0.2, stratify=Y_categorical, random_state=123)

#%% Embedding
from keras.models import Sequential
from keras.layers import Embedding, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

vocabulario = len(tokenizer.word_index) + 1  # Tamaño del vocabulario
embedding_dim = 20  # Dimensión del embedding # Si es para 3 GRD hay que cambiar el valor, porque 

embedding_model = Sequential()
embedding_model.add(Embedding(input_dim=vocabulario, output_dim=embedding_dim, input_length=x_train.shape[1]))
embedding_model.add(Flatten())

#%%
classes = GRD_comunes 
#classes = GRD_top20
x_train_embedded = embedding_model.predict(x_train)
x_val_embedded = embedding_model.predict(x_val)

def plot_confusion_matrix(y_true, y_pred,classes , model_name):
    cm = confusion_matrix(y_true, y_pred)
    Size = 30
    plt.figure(figsize=(15,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels = classes , yticklabels = classes, annot_kws={"size": Size})
    plt.title(f'Confusion Matrix para {model_name}', fontsize = Size)
    plt.xlabel('Predicted',fontsize = Size)
    plt.xticks(fontsize = Size/2)
    plt.yticks(fontsize = Size/2)
    plt.ylabel('Actual',fontsize = Size)
    plt.show()
    
print(len(np.unique(Y)))

#%% SVM 

svm_model = SVC(kernel = 'rbf') # para cualquier GRD 
#svm_model = SVC(kernel = 'rbf', C = 5) # para GRD_otros 
svm_model.fit(x_train_embedded, y_train)
y_pred_svm = svm_model.predict(x_val_embedded)
accuracy_svm = accuracy_score(y_val, y_pred_svm)
y_train_pred_svm = svm_model.predict(x_train_embedded)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print(f"SVM Training Accuracy: {round(train_accuracy_svm*100,2)}")
print(f"SVM Accuracy: {round(accuracy_svm*100,2)}")

plot_confusion_matrix(y_val, y_pred_svm, classes , f'SVM para {len(np.unique(Y))} GRD ')


#%%  Regresión Logística


lr_model = LogisticRegression(max_iter=1000, C = 10 , solver = 'saga') 
lr_model.fit(x_train_embedded, y_train)
y_pred_lr = lr_model.predict(x_val_embedded)
accuracy_lr = accuracy_score(y_val, y_pred_lr)
y_train_pred_lr = lr_model.predict(x_train_embedded)
train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
print(f"LR Training Accuracy: {round(train_accuracy_lr*100,2)}")
print(f"Logistic Regression Accuracy: {round(accuracy_lr*100,2)}")
plot_confusion_matrix(y_val, y_pred_lr, classes , f'LR para {len(np.unique(Y))} GRD ')


#%%  Random Forest

rf_model = RandomForestClassifier(n_estimators= 200)
rf_model.fit(x_train_embedded, y_train)
y_pred_rf = rf_model.predict(x_val_embedded)
accuracy_rf = accuracy_score(y_val, y_pred_rf)
y_train_pred_rf = rf_model.predict(x_train_embedded)
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
print(f"RF Training Accuracy: {round(train_accuracy_rf*100,2)}")
print(f"Random Forest Accuracy: {round(accuracy_rf*100,2)}")
plot_confusion_matrix(y_val, y_pred_lr, classes , f'RF para {len(np.unique(Y))} GRD ')

