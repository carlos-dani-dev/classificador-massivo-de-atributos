import sys
import cv2 as cv
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL as pil
import tensorflow as tf
from tqdm import tqdm
from functools import wraps
from keras_facenet import FaceNet

from facenet_pytorch import MTCNN
import torch

from concurrent.futures import ThreadPoolExecutor


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Pytorch usando dispositivo: {device}")
MTCNN_MODEL = MTCNN(keep_all=True, device=device)

PATH = "./img_align_celeba/img_align_celeba/"
ANNOTATION = pd.read_csv("./list_attr_celeba.csv")
ANNOTATION.set_index("image_id", inplace=True)

TOTAL_IMAGES = len(ANNOTATION)

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


def load_single_image(filename):
    img = cv.imread(filename)
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    img_resized = cv.resize(img_rgb, (720, 720))
    
    return Image.fromarray(img_resized)
    
    #return cv.resize(pil.Image.open(filename).convert("RGB"), (720, 720))


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
    """"
    Redimensiona o array para o tamanho de entrada do modelo
    """
    
    cropped_face_image = pil.Image.fromarray(cropped_face_array)
    face_image = cropped_face_image.convert('RGB')
    resized_face_image = face_image.resize(required_size)
    
    return np.asarray(resized_face_image)


@capture_output
def get_embeddings(facenet, resized_image_array_batch):

    embedding = facenet.embeddings(resized_image_array_batch)
    return embedding


def load_faces_generator(directory, batch_size=128):
    
    image_filenames_batch = []
    image_annotations_batch = []

    for image_file in os.listdir(directory):
        image_filename = directory + image_file
        image_annotation = ANNOTATION.loc[image_file]
        
        image_filenames_batch.append(image_filename)
        image_annotations_batch.append(image_annotation.to_dict())

        if len(image_filenames_batch) == batch_size:
            yield image_filenames_batch, image_annotations_batch
            image_filenames_batch=[]
            image_annotations_batch=[]
    
    if len(image_filenames_batch) > 0:
        yield image_filenames_batch, image_annotations_batch


def load_face_embeddings():
    facenet = FaceNet()

    chaves = ["Filename", "Face_embedding",
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald","Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
        "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
        "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
        "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face",
        "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
        "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
        "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]

    df_dict = {chave: [] for chave in chaves}

    pbar = tqdm(total=TOTAL_IMAGES, desc="Processamento de imagens em lotes", ncols=100)
    with ThreadPoolExecutor(max_workers=18) as executor:
        for image_filenames_batch, image_annotations_batch in load_faces_generator(PATH):
            
            resized_image_array_batch = []
            valid_index = []

            original_images_batch = list(executor.map(load_single_image, image_filenames_batch))
            boxes_batch, probs_batch = MTCNN_MODEL.detect(original_images_batch)

            for i in range(len(original_images_batch)):
                img_boxes = boxes_batch[i] if boxes_batch is not None else None
                img_probs = probs_batch[i] if probs_batch is not None else None

                cropped_image_array = crop_face(original_images_batch[i], img_boxes, img_probs)
                if cropped_image_array is None: continue
                
                resized_image_array = redim_face(cropped_image_array)
                resized_image_array_batch.append(resized_image_array)
                valid_index.append(i)
            
            if not resized_image_array_batch:
                pbar.update(len(image_filenames_batch))
                continue
                
            image_embeddings_batch=get_embeddings(facenet, resized_image_array_batch)

            for idx, i in enumerate(valid_index):
                df_dict["Filename"].append(image_filenames_batch[i])
                df_dict["Face_embedding"].append(image_embeddings_batch[idx])

                for attribute in image_annotations_batch[i].keys():
                    df_dict[attribute].append(image_annotations_batch[i][attribute])

            pbar.update(len(image_filenames_batch))

    pbar.close()

    df = pd.DataFrame(df_dict)
    df.to_pickle("../../proc/proc_celeba/embeddings_celeba.pkl")

if __name__ == "__main__":

    load_face_embeddings()