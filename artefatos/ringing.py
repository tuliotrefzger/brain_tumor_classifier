import cv2
import numpy as np


def to_0255(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)


def create_circular_mask(h, w, r):
    # Mask with center circle as 1, remaining as zeros
    # h,w = image.shape
    # r = circle radius
    ch, cw = h // 2, w // 2  # center coordinates
    y, x = np.ogrid[-ch : h - ch, -cw : w - cw]
    boolmask = x * x + y * y <= r * r
    mask = np.zeros((h, w, 2), np.uint8)
    mask[boolmask] = 1
    return mask


def fourier_apply_mask(im, mask):
    # Converting image to frequency domain
    dft = cv2.dft(np.float64(im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    fshift = dft_shift * mask  # apply mask
    f_ishift = np.fft.ifftshift(fshift)  # inverse shift
    imfilt = cv2.idft(f_ishift)  # inverse dft
    im_final = cv2.magnitude(
        imfilt[:, :, 0], imfilt[:, :, 1]
    )  # merge real and imaginary parts
    return im_final


# Primeira versão
# def generate_ringing(img, deg_level):
#     # deg_level = 1 significa pouco ringing e 10 significa muito
#     # Normaliza deg_level em uma escala de 40 a 120 para ser o raio do filtro
#     radius = np.uint((((deg_level-11)*-1)-1) / 9 * 80 + 40)
#     h, w = img.shape
#     mask = create_circular_mask(h, w, radius)
#     img_f = fourier_apply_mask(img, mask) #img filtrada sem escala com floats
#     img0255 = to_0255(img_f) # img 0255 int
#     return img0255

# Segunda versão
def generate_ringing(img, deg_level):
    # deg_level = 1 significa pouco ringing e 10 significa muito
    # Normaliza deg_level em uma escala de 50 a 180 para ser o raio do filtro
    radius = np.uint((((deg_level - 11) * -1) - 1) / 9 * 50 + 16)
    h, w = img.shape
    mask = create_circular_mask(h, w, radius)
    img_f = fourier_apply_mask(img, mask)  # img filtrada sem escala com floats
    img0255 = to_0255(img_f)  # img 0255 int
    return img0255
