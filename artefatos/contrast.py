import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_transform(img, alpha, beta):  
    new_img = np.zeros(img.shape, img.dtype)

    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         new_img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
    
    # removi o loop que estava causando a lentidão enorme no processamento
    new_img = np.clip(alpha * img + beta, 0, 255)


    return new_img.astype(np.uint8)

def generate_contrast(img, deg_level):
    alpha = (11-deg_level)*0.09              # Normaliza deg_level em uma escala de 0.09 a 0.90 para ser o alfa da transformação
    beta = 128*(1-alpha)-1              # beta é definido de forma a manter o histograma no centro (não gerar imagens apenas mais escuras)

    new_img = linear_transform(img, alpha, beta)  # Aplica a transformação linear à imagem

    return new_img.astype(np.uint8)