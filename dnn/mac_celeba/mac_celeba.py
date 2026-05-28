import os
import math
import cv2 as cv
import PIL as pil
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from keras import Model, activations
import sklearn
from sklearn.model_selection import train_test_split
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import Accuracy
#from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import ReLU,LeakyReLU, Softmax, BatchNormalization, Input, Dense, Dropout
#from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())


DF_CELEBA = pd.read_pickle('../../data_prep/proc/proc_celeba/male_balanced_embeddings_celeba.pkl')
DF_CELEBA = DF_CELEBA[['Filename', 'Face_embedding', 'Male']]
ATRIBUTOS = DF_CELEBA.columns[2:]

EPOCHS = 200
BATCH_SIZE = 1024


def load_embeddings():

    DF_CELEBA["Male"] = DF_CELEBA["Male"].replace(-1, 0)

    celeba_embeddings = np.vstack(DF_CELEBA['Face_embedding'].values)
    celeba_labels = np.vstack(DF_CELEBA['Male'].values)

    print(len(celeba_embeddings), ' de shape ', celeba_embeddings.shape)
    print(len(celeba_labels), ' de shape ', celeba_labels.shape)
    
    return celeba_embeddings, celeba_labels


def manipulating_embeddings(celebA_embeddings, celebA_labels):
    """
    Separa os embeddings em subsets de treinamento, teste e validação. Também corrige o formato
        dos labels para entrada no multi-task larning model

    Args:
        celebA_embeddings (np.array): Um array 1D com os embeddings calculados do dataset (ordenados)
        celebA_labels (np.array): Um array 1D com os labels dos embeddings calculados do dataset (ordenados)

    Returns:
        list: Lista com 3 tuplas de 2 valores cad:  treinamento/teste/validação (ordenados) e seus respectivos
            labels (ordenados) 
    """

    # print('Shape anterior: ', celebA_embeddings.shape)
    celebA_embeddings = np.reshape(celebA_embeddings, (-1, 512,))
    # print('Novo shape: ', celebA_embeddings.shape)

    # for label in celebA_labels:
    #     for i in range(0, len(label)):
    #         if label[i] == -1: label[i] = 0

    # trainX, trainy = celebA_embeddings[0:70000], celebA_labels[0:70000]
    # testX, testy = celebA_embeddings[70000:85000], celebA_labels[70000:85000]
    # validX, validy = celebA_embeddings[85000:100000], celebA_labels[85000:100000]


    X_train, X_test, y_train, y_test = train_test_split(celebA_embeddings, celebA_labels,
                                                        test_size=0.30, random_state=42)

    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,
                                                        test_size=0.50, random_state=42)

    original_labels = [y_train, y_test, y_valid]  # Lista de arrays
    output_labels = []

    for labels in original_labels:
        branch_labels = []

        labels_transposed = np.array(labels).T
        for i in range(len(ATRIBUTOS)):
            branch_labels.append(labels_transposed[i])

        output_labels.append(branch_labels)  
    
    return [(X_train, output_labels[0]), (X_test, output_labels[1]), (X_valid, output_labels[2])]


def subnet(shared_layers_output, i):
    """
    Define as subredes do multi-task learning model

    Args:
        shared_layers_output (): A saída da camada compartilhada imediatamente anterior
        i (int): o número da subrede (nesse caso, 0-39, pois são 40 atributos)
 
    Returns:
        double : A saída de cada função sigmoid da subrede
    """
    
    att_branch = Dense(512, name='dense_'+str(i)+'_1')(shared_layers_output)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.5)(att_branch)

    att_branch = Dense(512, name='dense_'+str(i)+'_2')(att_branch)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.5)(att_branch)

    branch_output = Dense(1, name=ATRIBUTOS[i], activation='sigmoid')(att_branch)

    return branch_output


def multitask_model_definition():
    """
    Define as camadas compartilhadas do multi-task learning model

    Args:
        attributes_qtt (int): A quantidade de atributos (e consequentemente de subredes)
 
    Returns:
        model: O modelo pronto para compilação e treinamento
    """

    input_layer = Input(shape=(512,), name='input_layer')
    
    shared_x = Dense(512, name='shared_dense_layer')(input_layer)
    shared_x = ReLU()(shared_x)
    shared_x = BatchNormalization()(shared_x)
    shared_x = Dropout(0.5)(shared_x)

    branch_outputs = list()
    for i in range(len(ATRIBUTOS)):
        branch_outputs.append(subnet(shared_x, i))

    model = Model(input_layer, branch_outputs, name='model')

    return model


def compiling_model(model):
    """
    Compila o multi-task learning model

    Args: 
        model (): O multi-task learning model definido
        attributes_qtt (int): A quantidade de atributos (e consequentemente de subredes)
 
    Returns:
        
    """

    epochs = EPOCHS
    initial_learning_rate = 1e-03
    decay_rate = initial_learning_rate/epochs
    decay_steps = 100000

    metrics_dict = {ATRIBUTOS[i]:['Accuracy'] for i in range(len(ATRIBUTOS))}
    loss_dict = {ATRIBUTOS[i]:'binary_crossentropy' for i in range(len(ATRIBUTOS))}

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,  # Número de passos antes de aplicar o decaimento
    decay_rate=decay_rate,   # Fator de decaimento ajustado
    staircase=True     # Decaimento em degraus
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=opt,
        loss=loss_dict,
        loss_weights=[1/len(ATRIBUTOS)]*len(ATRIBUTOS),
        metrics=metrics_dict
    )


def training_model(model, X_train, X_test, y_train_input, y_test_input):
    """
    Treina o multi-task learning model, uma vez já compilado

    Args:
        model: O multi-task learning model
        trainX (np.array): O np.array com os embeddings de treinamento
        testX (np.array): O np.array com os embeddings de teste
        trainy_input (list): A lista com os labels de formato corrigido
        testy_input (list): A lista com os labels de formato corrigido
        batch_size (int): O batch size para treinamento do modelo
        epochs (int): A quantidade de épocas para treinamento do modelo

    Returns:
        history: o history de treinamento do modelo
    """ 

    print("Shape de X_train:", getattr(X_train, 'shape', 'Não tem shape (pode ser lista)'))
    print("Shape de X_test:", getattr(X_test, 'shape', 'Não tem shape (pode ser lista)'))
    print("Shape de y_train_input:", getattr(y_train_input, 'shape', 'Não tem shape (pode ser lista)'))
    print("Shape de y_test_input:", getattr(y_test_input, 'shape', 'Não tem shape (pode ser lista)'))
    
    history = model.fit(
    x=X_train,
    y=y_train_input,
    validation_data=(X_test, y_test_input),
    batch_size=BATCH_SIZE,
    verbose=2,
    epochs=EPOCHS
#   callbacks=[tensorboard_callback]
    )

    model_name = 'modelo_celebA_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+'.keras'
    model.save(model_name)

    return history


celeba_embeddings, celeba_labels = load_embeddings()
(X_train, y_train_input), (X_test, y_test_input), (X_valid, Y_valid_input) = manipulating_embeddings(celeba_embeddings, celeba_labels)
model = multitask_model_definition()

compiling_model(model)

history = training_model(model, X_train, X_test, y_train_input, y_test_input)
history_df = pd.DataFrame(history.history)
history_df.to_csv('modelo_celebA_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+'.csv')