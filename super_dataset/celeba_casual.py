from concurrent.futures import ThreadPoolExecutor
import os
import warnings

from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import io
import sys
import cv2 as cv
import pandas as pd
import numpy as np
from functools import wraps
from keras_facenet import FaceNet
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = 'error'
warnings.filterwarnings("ignore")

device = 'cuda'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for phy_device in physical_devices:
        tf.config.experimental.set_memory_growth(phy_device, True)

MODEL_CASUAL = load_model("../dnn/mac_casual/modelo_cc_200_4096_1e-05.keras")
MODEL_CELEBA = load_model("../dnn/mac_celeba/modelo_celebA_200_1024.keras")
DF_CASUAL = pd.read_pickle("../data_prep/proc/proc_casual/fitz_type_balanced_embeddings_casual.pkl")
DF_CELEBA = pd.read_pickle("../data_prep/proc/proc_celeba/male_balanced_embeddings_celeba.pkl")
ATRIBUTOS_CASUAL = DF_CASUAL.columns[1:]
# ATRIBUTOS_CELEBA = DF_CELEBA.columns[2:]
ATRIBUTOS_CELEBA = ['male']

DATASET_NAME = "celeba"

LOADED_LABEL_ENCODERS = {}
with open("../dnn/mac_casual/label_encoders.pkl", 'rb') as f:
    LOADED_LABEL_ENCODERS = pickle.load(f)

for layer in MODEL_CASUAL.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

for layer in MODEL_CELEBA.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print(f"[INFO] Pytorch usando dispositivo: {device}")
MTCNN_MODEL = MTCNN(keep_all=True, device=device)
FACENET = FaceNet()

# LOADED_LABEL_ENCODERS = {}
# with open('./files/label_encoders.pkl', 'rb') as f:
#     LOADED_LABEL_ENCODERS = pickle.load(f)


def capture_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper


@capture_output
def get_embeddings(img):
    return FACENET.embeddings(img)


@capture_output
def get_embeddings_batch(img_batch):
    embeddings_batch = FACENET.embeddings(img_batch)
    return np.reshape(embeddings_batch, (-1, 512))

@capture_output
def crop_face(original_image, boxes, probs):
    """
    Extrai a primeira face detectada usando MTCNN (PyTorch).
    Retorna o rosto recortado e redimensionado como array RGB.
    """

    if boxes is not None and len(boxes) > 0:
        alpha=40
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = [int(coord) for coord in boxes[best_idx]]

        x1-=alpha
        y1-=alpha
        x2+=alpha
        y2+=alpha

        w, h = original_image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped_face = original_image.crop((x1, y1, x2, y2))

        return np.asarray(cropped_face)
    else:
        return None  # Nenhuma face detectada


def redim_face(cropped_face_array, required_size=(160, 160)):
    
    cropped_face_image = Image.fromarray(cropped_face_array)
    face_image = cropped_face_image.convert('RGB')
    resized_face_image = face_image.resize(required_size)
    
    return np.asarray(resized_face_image)


def extract_face_batch(img_batch, req_size=(160, 160)):

    img_rgb_batch = [i.convert('RGB') for i in img_batch]

    boxes_batch, probs_batch = MTCNN_MODEL.detect(img_rgb_batch)

    detected_faces_idx = []

    img_array_batch = []

    for i in range(len(img_rgb_batch)):
        img_boxes = boxes_batch[i]
        img_probs = probs_batch[i]

        cropped_image_array = crop_face(img_rgb_batch[i], img_boxes, img_probs)
        if cropped_image_array is None: continue
        
        resized_image_array = redim_face(cropped_image_array)
        img_array_batch.append(resized_image_array)
        detected_faces_idx.append(i)
    
    return img_array_batch, detected_faces_idx


def extract_face(frame, required_size=(160, 160)):
    if isinstance(frame, np.ndarray):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
    elif isinstance(frame, Image.Image):
        img = frame.convert('RGB')
    else:
        raise ValueError("Entrada inválida: deve ser numpy array (cv2) ou PIL.Image")

    boxes, probs = MTCNN.detect(img)

    if boxes is not None and len(boxes) > 0:
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = [int(coord) for coord in boxes[best_idx]]

        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img.crop((x1, y1, x2, y2))
        face_resized = face.resize(required_size)
        return np.asarray(face_resized)
    else:
        return None


def decode_image_prediction(predictions):
    decoded_predictions = []
    for i, probs in enumerate(predictions):
        if tf.is_tensor(probs):
            probs = probs.numpy()

        probs = np.squeeze(probs)

        if probs.ndim == 0 or (probs.ndim == 1 and probs.shape[0] == 1):
            predicted_class_index = (probs > 0.5).astype(int)
            decoded_label = LOADED_LABEL_ENCODERS[i].inverse_transform([predicted_class_index])[0]
        else:
            predicted_class_index = np.argmax(probs, axis=-1)
            decoded_label = LOADED_LABEL_ENCODERS[i].inverse_transform([predicted_class_index])[0]

        decoded_predictions.append(decoded_label)

    return np.array(decoded_predictions).reshape(1, -1)


def calculate_reliability(probs_array, alpha=0.05):
    x = np.array(probs_array, dtype=np.float32)
    centrality = (1.0 - alpha) * np.mean(x)

    diff_matrix = np.abs(x[:, None] - x[None, :])
    dispersion = alpha * np.mean(diff_matrix)

    reliability = centrality - dispersion
    return round(float(reliability), 5)


def inference_batch(embeddings_batch, atributos, model):

    dict_batch = []

    pred_batch = model(embeddings_batch, training=False)
    pred_batch_np = tf.squeeze(pred_batch).numpy()

    for idx in range(len(embeddings_batch)):

        dict_ = {}

        total_atributos = len(atributos)
        for j in range(total_atributos):
            if total_atributos == 1:
                probs = pred_batch_np[idx]
            else:
                probs = pred_batch_np[j, idx]

            if tf.is_tensor(probs):
                probs = probs.numpy()

            #probs = np.squeeze(probs)

            predicted_class_index=-1

            if probs.ndim == 0 or probs.shape == ():

                predicted_class_index = (probs > 0.5).astype(int)
                
                prob_value = float(probs)
                idx_max = 0 if prob_value <= 0.5 else 1
            elif probs.ndim == 1 and probs.shape[0] == 1:
                
                predicted_class_index = (probs > 0.5).astype(int)
                
                prob_value = float(probs[0])
                idx_max = 0 if prob_value <= 0.5 else 1
            else:

                predicted_class_index = np.argmax(probs, axis=-1)

                idx_max = np.argmax(probs)
                prob_value = float(probs[idx_max])
            
            decoded_label=None
            if DATASET_NAME == "celeba":
                decoded_label = LOADED_LABEL_ENCODERS[j].inverse_transform([predicted_class_index])[0]
            elif DATASET_NAME == "casual":
                decoded_label = "male" if probs > 0.5 else "not male"

            dict_[f"{atributos[j]}_prediction"] = decoded_label
            dict_[f"{atributos[j]}_inference_idx"] = idx_max
            dict_[f"{atributos[j]}_reliability"] = -1
        dict_batch.append(dict_)

    return dict_batch


def model_inference_function(img, model):
    
    face_img = extract_face(img, required_size=(160, 160))
    if face_img is None:
        return None, None
        
    face_img = np.expand_dims(face_img, axis=0)
    
    img_embedding = get_embeddings(face_img)
    img_embedding = np.reshape(img_embedding, (1, 512))
    
    model_inference_prediction = model(img_embedding, training=False)
    decoded_inference_prediction = decode_image_prediction(model_inference_prediction)
    
    dict_inference = {a: {} for a in ATRIBUTOS_CC}

    for i in range(len(ATRIBUTOS_CC)):
        dict_ = {}
        probs = model_inference_prediction[i]

        if tf.is_tensor(probs):
            probs = probs.numpy()

        probs = np.squeeze(probs)

        if probs.ndim == 0 or probs.shape == ():  
            prob_value = float(probs)
            idx_max = 0 if prob_value <= 0.5 else 1
        elif probs.ndim == 1 and probs.shape[0] == 1:
            prob_value = float(probs[0])
            idx_max = 0 if prob_value <= 0.5 else 1
        else:
            idx_max = np.argmax(probs)
            prob_value = float(probs[idx_max])

        dict_[f"{ATRIBUTOS_CC[i]}_decoded_pred"] = decoded_inference_prediction[0][i]
        dict_[f"{ATRIBUTOS_CC[i]}_decoded_index"] = idx_max
        dict_inference[f"{ATRIBUTOS_CC[i]}"] = dict_

    return dict_inference, img_embedding


def batch_model_training_function(model, img_embedding, m):

    img_embedding = np.reshape(img_embedding, (1, 512))
    img_embedding_batch = np.repeat(img_embedding, m, axis=0)

    model_training_prediction_batch = model(img_embedding_batch, training=True)

    dict_training = {}
    for i in range(len(atributos)): 
        probs = model_training_prediction_batch[i]
        if tf.is_tensor(probs):
            probs = probs.numpy()
        dict_training[atributos[i]] = np.squeeze(probs)

    return dict_training


def load_faces_generator(df, batch_size=128):
    
    img_embeddings = df['Face_embedding'].values if DATASET_NAME == "celeba" else df["face embedding"].values

    img_embeddings = np.vstack(img_embeddings)

    for i in range(0, len(img_embeddings), batch_size):
            yield img_embeddings[i:i+batch_size]


def load_single_image(filename):
    return Image.open(filename).convert("RGB")

def prediction_function(df, model, atributos, m=100):

    for img_embedding_batch in load_faces_generator(df, batch_size=128):

        list_dict = []

        dict_inference_batch = inference_batch(img_embedding_batch, atributos, model)

        img_embedding_giant_batch = np.repeat(img_embedding_batch, m, axis=0)
        training_giant_batch = model(img_embedding_giant_batch, training=True)

        #numpy_training_giant_batch = np.array([out.numpy() if tf.is_tensor(out) else out for out in training_giant_batch], dtype=np.float32)

        for i in range(len(img_embedding_batch)):

            current_face_dict = dict_inference_batch[i]
            current_face_embedding = img_embedding_batch[i]

            #batch_dict_training = batch_model_training_function(model, current_face_embedding, m)

            for j, atributo in enumerate(atributos):
                #raw_batch_predictions = batch_dict_training[atributo]
                idx_target = current_face_dict[f"{atributo}_inference_idx"]

                training_pred_start_idx = i*m
                training_pred_end_idx = training_pred_start_idx+m

                # print("inicio: ", training_pred_start_idx)
                # print("fim: ",training_pred_end_idx)
                # print(training_giant_batch.shape)
                probs = training_giant_batch[training_pred_start_idx:training_pred_end_idx]
                # print("PROBS: ", probs)
                if tf.is_tensor(probs):
                    probs = probs.numpy()

                raw_batch_predictions = probs
                if raw_batch_predictions.ndim == 1 or (raw_batch_predictions.ndim == 2 and raw_batch_predictions.shape[1] == 1):
                    raw_batch_predictions = np.squeeze(raw_batch_predictions)
                    if idx_target == 0:
                        lista_100_predicoes = 1.0 - raw_batch_predictions
                    else:
                        lista_100_predicoes = raw_batch_predictions
                else:
                    lista_100_predicoes = raw_batch_predictions[:, idx_target]
                
                confiabilidade = calculate_reliability(lista_100_predicoes, alpha=0.05)
                
                current_face_dict[f"{atributo}_reliability"] = confiabilidade

            list_dict.append(current_face_dict)
        
        if list_dict:
            yield list_dict


if __name__ == "__main__":
    
    df = None
    model = None
    atributos = None



    if DATASET_NAME == "celeba":
        df = DF_CELEBA
        model = MODEL_CASUAL # se vou anotar um, tem que ser com os atributos do outro
        atributos = ATRIBUTOS_CASUAL
    if DATASET_NAME == "casual":
        df = DF_CASUAL
        model = MODEL_CELEBA # se vou anotar um, tem que ser com os atributos do outro
        atributos = ATRIBUTOS_CELEBA
    
    print('ATRIBUTOS: ', atributos)

    list_pred = []
    total_imgs = len(df)
    
    pbar = tqdm(total=total_imgs, desc="Processamento de imagens em lotes", ncols=100)
    for pred_batch in prediction_function(df, model, atributos, m=100):
        
        list_pred.extend(pred_batch)

        pbar.update(len(pred_batch))

    pbar.close()
    
    print(f"DATASET {DATASET_NAME} COMPLETAMENTE ANOTADO")

    df_ = pd.DataFrame(list_pred)
    if DATASET_NAME == "celeba":
        DF_CELEBA['fitz_type'] = df_['fitz_type_prediction'].values
        DF_CELEBA.to_pickle(f"{DATASET_NAME}_annotated.pkl")
    else:
        DF_CASUAL['male'] = df_['male_prediction'].values
        DF_CASUAL.to_pickle(f"{DATASET_NAME}_annotated.pkl")