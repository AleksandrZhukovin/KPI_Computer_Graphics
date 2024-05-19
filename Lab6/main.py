import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image = cv2.imread("images/InputImage.png")
wm = cv2.imread('images/WM.png')

wm_blue = wm[:, :, 2]

blue_chanel = image[:, :, 2]

wm_bite = np.zeros(wm.shape[0:2], dtype=int)

for i in range(wm.shape[0]):
    for j in range(wm.shape[1]):
        wm_bite[i, j] = int('{0:08b}'.format(wm_blue[i, j])[0])

binary_1 = np.zeros(blue_chanel.shape, dtype=int)
binary_2 = np.zeros(blue_chanel.shape, dtype=int)
binary_3 = np.zeros(blue_chanel.shape, dtype=int)
binary_4 = np.zeros(blue_chanel.shape, dtype=int)
binary_5 = np.zeros(blue_chanel.shape, dtype=int)
binary_6 = np.zeros(blue_chanel.shape, dtype=int)
binary_7 = np.zeros(blue_chanel.shape, dtype=int)
binary_8 = np.zeros(blue_chanel.shape, dtype=int)

for i in range(blue_chanel.shape[0]):
    for j in range(blue_chanel.shape[1]):
        number = '{0:08b}'.format(blue_chanel[i, j])
        binary_1[i, j] = int(number[7])
        binary_2[i, j] = int(number[6])
        binary_3[i, j] = int(number[5])
        binary_4[i, j] = int(number[4])
        binary_5[i, j] = int(number[3])
        binary_6[i, j] = int(number[2])
        binary_7[i, j] = int(number[1])
        binary_8[i, j] = int(number[0])

result_1 = np.zeros(blue_chanel.shape)
result_2 = np.zeros(blue_chanel.shape)
result_3 = np.zeros(blue_chanel.shape)
result_4 = np.zeros(blue_chanel.shape)
result_5 = np.zeros(blue_chanel.shape)
result_6 = np.zeros(blue_chanel.shape)
result_7 = np.zeros(blue_chanel.shape)
result_8 = np.zeros(blue_chanel.shape)

for i in range(blue_chanel.shape[0]):
    for j in range(blue_chanel.shape[1]):
        number_1 = (str(wm_bite[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_1 = int(number_1, 2)
        result_1[i, j] = number_1

        number_2 = (str(binary_1[i, j]) + str(wm_bite[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_2 = int(number_2, 2)
        result_2[i, j] = number_2

        number_3 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(wm_bite[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_3 = int(number_3, 2)
        result_3[i, j] = number_3

        number_4 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(wm_bite[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_4 = int(number_4, 2)
        result_4[i, j] = number_4

        number_5 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(wm_bite[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_5 = int(number_5, 2)
        result_5[i, j] = number_5

        number_6 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(wm_bite[i, j]) + str(binary_7[i, j]) + str(binary_8[i, j]))[::-1]
        number_6 = int(number_6, 2)
        result_6[i, j] = number_6

        number_7 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(wm_bite[i, j]) + str(binary_8[i, j]))[::-1]
        number_7 = int(number_7, 2)
        result_7[i, j] = number_7

        number_8 = (str(binary_1[i, j]) + str(binary_2[i, j]) + str(binary_3[i, j]) + str(binary_4[i, j]) +
                    str(binary_5[i, j]) + str(binary_6[i, j]) + str(binary_7[i, j]) + str(wm_bite[i, j]))[::-1]
        number_8 = int(number_8, 2)
        result_8[i, j] = number_8

image_wm_1 = image.copy()
image_wm_2 = image.copy()
image_wm_3 = image.copy()
image_wm_4 = image.copy()
image_wm_5 = image.copy()
image_wm_6 = image.copy()
image_wm_7 = image.copy()
image_wm_8 = image.copy()

image_wm_1[:, :, 2] = result_1
image_wm_2[:, :, 2] = result_2
image_wm_3[:, :, 2] = result_3
image_wm_4[:, :, 2] = result_4
image_wm_5[:, :, 2] = result_5
image_wm_6[:, :, 2] = result_6
image_wm_7[:, :, 2] = result_7
image_wm_8[:, :, 2] = result_8

# im = Image.fromarray(image_wm_1)
# im.save("images/image_wm_1.png")
# im = Image.fromarray(image_wm_2)
# im.save("images/image_wm_2.png")
# im = Image.fromarray(image_wm_3)
# im.save("images/image_wm_3.png")
# im = Image.fromarray(image_wm_4)
# im.save("images/image_wm_4.png")
# im = Image.fromarray(image_wm_5)
# im.save("images/image_wm_5.png")
# im = Image.fromarray(image_wm_6)
# im.save("images/image_wm_6.png")
# im = Image.fromarray(image_wm_7)
# im.save("images/image_wm_7.png")
# im = Image.fromarray(image_wm_8)
# im.save("images/image_wm_8.png")


image_with_wm = cv2.imread('images/image_wm_8.png')

blue_chanel = image_with_wm[:, :, 2]

for i in range(blue_chanel.shape[0]):
    for j in range(blue_chanel.shape[1]):
        number = '{0:08b}'.format(blue_chanel[i, j])
        number = number[1] + number[1:]
        blue_chanel[i, j] = int(number, 2)

image_with_wm[:, :, 2] = blue_chanel

# im = Image.fromarray(image_with_wm)
# im.save("images/removed.png")


