# Necessary:
# pip install torchio
from PIL import Image
import numpy as np
import torchio as tio

def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)

def pil_to_tio(img):
    npimg = np.array(img)
    # A biblioteca torchio precisa dos channels na primeira dimensão
    npimgcwh = np.transpose(npimg,(2, 0, 1))
    # Adicionando mais uma dimensão para ficar 4d
    npimg4d = npimgcwh[..., np.newaxis]
    tioimg = tio.ScalarImage(tensor=npimg4d)
    return tioimg

def tio_to_pil(img4d):
    npimgcwh = np.squeeze(img4d)
    # Voltando o channel para última dimensão
    npimg = np.transpose(npimgcwh, (1, 2, 0))
    npimg255 = to_0255(npimg)
    img = Image.fromarray(npimg255)
    return img

def np_1ch_to_pil(img):
    pimg = np.repeat(img[..., np.newaxis], 3, -1) # converte para 3 canais
    pimg = Image.fromarray(pimg)
    return pimg

def pil_to_np_1ch(img):
    npimg = np.array(img)
    npimg = np.dot(npimg, [0.299, 0.587, 0.114]).astype(np.uint8)
    return npimg

def generate_ghosting(img, deg_level):
    # deg_level = 1 significa pouco ringing e 10 significa muito
    intensities = np.linspace(0.3,1.1,10)
    pil_img = np_1ch_to_pil(img)
    pio_img = pil_to_tio(pil_img)
    func = tio.Ghosting(num_ghosts=5 , axis=0, intensity=intensities[deg_level-1], restore=0.02)
    new_pio_img = func(pio_img)
    new_pil_img = tio_to_pil(new_pio_img)
    new_np_img = pil_to_np_1ch(new_pil_img)
    return new_np_img