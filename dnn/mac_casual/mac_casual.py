import os
import math
import pickle
import numpy as np
import pandas as pd
from keras import Model
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import ReLU, BatchNormalization, Input, Dense, Dropout


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

DF_CASUAL = pd.read_pickle('../../data_prep/proc/proc_casual/fitz_type_balanced_embeddings_casual.pkl')
DF_CASUAL = DF_CASUAL[['face embedding', 'fitz_type']]
print(DF_CASUAL.columns)
ATRIBUTOS = DF_CASUAL.columns[1:]

LABEL_ENCODERS = {}
EPOCHS = 200
BATCH_SIZE = 4096
INITIAL_LR = 1e-005


def load_embeddings():
    
    cc_embeddings = np.vstack(DF_CASUAL["face embedding"])
    cc_labels = np.vstack(DF_CASUAL["fitz_type"])
    
    return cc_embeddings, cc_labels


def output_form(original_label):
    return [np.array(label) for label in original_label]


def manipulating_embeddings(cc_embeddings, cc_labels):
    cc_embeddings = np.reshape(cc_embeddings, (-1, 512, ))
    # cc_labels[cc_labels == -1] = 0

    # cc_embeddings, cc_labels = shuffle(cc_embeddings, cc_labels, random_state=42)

    # trainX = cc_embeddings[:135006]
    # trainy = cc_labels[:135006]

    # testX = cc_embeddings[135006:150006]
    # testy = cc_labels[135006:150006]

    X_train, X_test, y_train, y_test = train_test_split(cc_embeddings, cc_labels,
                                                        test_size=0.30, random_state=42)

    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,
                                                        test_size=0.50, random_state=42)

    label_encoders = {}
    y_train_input, y_test_input, y_valid_input = [], [], []

    for i in range(y_train.shape[1]):
        le = LabelEncoder()
        le.fit(y_train[:, i])

        train_col = le.transform(y_train[:, i])
        test_col  = le.transform(y_test[:, i])
        valid_col = le.transform(y_valid[:,i])

        num_classes = len(le.classes_)
        label_encoders[i] = le

        y_train_input.append(to_categorical(train_col, num_classes))
        y_test_input.append(to_categorical(test_col, num_classes))
        y_valid_input.append(to_categorical(valid_col, num_classes))

    global LABEL_ENCODERS
    LABEL_ENCODERS = label_encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(LABEL_ENCODERS, f)

    return (X_train, y_train_input), (X_test, y_test_input), (X_valid, y_valid_input)


def subnet(shared_layers_output, i):
        
    att_branch = Dense(512, name='dense_'+str(i)+'_1')(shared_layers_output)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.2)(att_branch)

    att_branch = Dense(512, name='dense_'+str(i)+'_2')(att_branch)
    att_branch = ReLU()(att_branch)
    att_branch = BatchNormalization()(att_branch)
    att_branch = Dropout(0.2)(att_branch)

    branch_output = None

    num_classes = len(LABEL_ENCODERS[i].classes_)

    if num_classes > 2:
        branch_output = Dense(num_classes, name=ATRIBUTOS[i], activation='softmax')(att_branch)
    else:
        branch_output = Dense(1, name=ATRIBUTOS[i], activation='sigmoid')(att_branch)

    return branch_output


def multitask_model_definition():
    
    input_layer = Input(shape=(512,), name='input_layer')
    
    shared_x = Dense(512, name='shared_dense_layer')(input_layer)
    shared_x = ReLU()(shared_x)
    shared_x = BatchNormalization()(shared_x)
    shared_x = Dropout(0.2)(shared_x)

    branch_outputs = list()
    for i in range(len(ATRIBUTOS)):
        branch_outputs.append(subnet(shared_x, i))

    model = Model(input_layer, branch_outputs, name='model')

    return model


def compiling_model(model):
    
    metrics_dict = {ATRIBUTOS[i]:['Accuracy'] for i in range(len(ATRIBUTOS))}
    loss_dict = {}
    
    for i in range(len(ATRIBUTOS)):
        if len(LABEL_ENCODERS[i].classes_) > 2:
            loss_dict[ATRIBUTOS[i]] = 'categorical_crossentropy'
        else:
            loss_dict[ATRIBUTOS[i]] = 'binary_crossentropy'

    epochs = EPOCHS
    initial_learning_rate = INITIAL_LR
    decay_rate = initial_learning_rate/epochs
    decay_steps = 10000

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
    )

    loss_weights = {}
    for i in range(len(ATRIBUTOS)):
        n_classes = len(LABEL_ENCODERS[i].classes_)
        loss_weights[ATRIBUTOS[i]] = 1.0 / math.log(n_classes + 1)

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=opt,
        loss=loss_dict,
        metrics=metrics_dict,
        loss_weights=loss_weights
    )


def training_model(model, trainX, testX, trainy_input, testy_input):

    history = model.fit(
        x=trainX,
        y=trainy_input,
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        verbose=2,
        epochs=EPOCHS
    )

    model_name = f'modelo_cc_{str(EPOCHS)}_{str(BATCH_SIZE)}_{str(INITIAL_LR)}.keras'
    model.save(model_name)

    return history


if __name__ == "__main__":

    cc_embeddings, cc_labels = load_embeddings()
    (X_train, y_train_input), (X_test, y_test_input), (X_valid, y_valid_input) = manipulating_embeddings(cc_embeddings, cc_labels)
    model = multitask_model_definition()
    tf.keras.utils.plot_model(model, to_file=f'modelo_cc_{str(EPOCHS)}_{str(BATCH_SIZE)}_{str(INITIAL_LR)}.png', show_shapes=True, show_layer_names=True)

    compiling_model(model)

    history = training_model(model, X_train, X_test, y_train_input, y_test_input)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'modelo_cc_{str(EPOCHS)}_{(BATCH_SIZE)}.csv')