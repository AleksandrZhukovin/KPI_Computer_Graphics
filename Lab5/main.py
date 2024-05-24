import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.fft import fft2, ifft2
from scipy.interpolate import RectBivariateSpline
from PIL import Image


file = open('num.csv', 'r')
image = np.array(file.readline().split(',')[1:], dtype=int).reshape((28, 28))
big_image = Image.open('big_image.jpg').convert('L')
big_image_arr = np.asarray(big_image)


def nearest_neighbor(img, new_size):
    old_height, old_width = image.shape
    new_height, new_width = new_size

    scale_y = old_height / new_height
    scale_x = old_width / new_width

    new_y, new_x = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing='ij')

    old_y = np.round(new_y * scale_y).astype(int)
    old_x = np.round(new_x * scale_x).astype(int)

    old_y = np.clip(old_y, 0, old_height - 1)
    old_x = np.clip(old_x, 0, old_width - 1)

    interpolated_image = img[old_y, old_x]

    return interpolated_image


def linear(img, new_size):
    old_height, old_width = img.shape
    new_height, new_width = new_size

    scale_y = old_height / new_height
    scale_x = old_width / new_width

    new_y, new_x = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing='ij')

    old_y = new_y * scale_y
    old_x = new_x * scale_x

    interpolated_image = np.zeros((new_height, new_width), dtype=img.dtype)
    for i in range(new_height):
        for j in range(new_width):
            y_floor = int(np.floor(old_y[i, j]))
            y_ceil = min(y_floor + 1, old_height - 1)
            x_floor = int(np.floor(old_x[i, j]))
            x_ceil = min(x_floor + 1, old_width - 1)

            interpolated_image[i, j] = (
                    (x_ceil - old_x[i, j]) * (y_ceil - old_y[i, j]) * img[y_floor, x_floor] +
                    (old_x[i, j] - x_floor) * (y_ceil - old_y[i, j]) * img[y_floor, x_ceil] +
                    (x_ceil - old_x[i, j]) * (old_y[i, j] - y_floor) * img[y_ceil, x_floor] +
                    (old_x[i, j] - x_floor) * (old_y[i, j] - y_floor) * img[y_ceil, x_ceil]
            )

    return interpolated_image


def polynomial_interpolation(img, new_size):
    old_height, old_width = img.shape
    new_height, new_width = new_size

    new_y, new_x = np.meshgrid(np.linspace(0, old_height - 1, new_height), np.linspace(0, old_width - 1, new_width), indexing='ij')

    f = interp2d(np.arange(old_width), np.arange(old_height), img, kind='cubic')
    interpolated_image = f(np.linspace(0, old_width - 1, new_width), np.linspace(0, old_height - 1, new_height))

    return interpolated_image


def fourier_interpolation(img, new_size):
    old_height, old_width = img.shape
    new_height, new_width = new_size

    f_image = fft2(img)

    u = np.fft.fftfreq(old_height)
    v = np.fft.fftfreq(old_width)

    U, V = np.meshgrid(u, v, indexing='ij')

    f_interpolated = np.zeros((new_height, new_width), dtype=np.complex128)
    f_interpolated[:old_height // 2, :old_width // 2] = f_image[:old_height // 2, :old_width // 2]
    f_interpolated[-old_height // 2:, :old_width // 2] = f_image[-old_height // 2:, :old_width // 2]
    f_interpolated[:old_height // 2, -old_width // 2:] = f_image[:old_height // 2, -old_width // 2:]
    f_interpolated[-old_height // 2:, -old_width // 2:] = f_image[-old_height // 2:, -old_width // 2:]

    interpolated_image = np.real(ifft2(f_interpolated))

    interpolated_image = interpolated_image[:new_height, :new_width]

    return interpolated_image


def spline_interpolation(img, new_size):

    old_height, old_width = img.shape
    new_height, new_width = new_size

    new_y, new_x = np.meshgrid(np.linspace(0, old_height - 1, new_height), np.linspace(0, old_width - 1, new_width), indexing='ij')

    spline = RectBivariateSpline(np.arange(old_height), np.arange(old_width), img)

    interpolated_image = spline.ev(new_y, new_x)

    return interpolated_image


def least_squares_interpolation(img, new_size):

    old_height, old_width = img.shape

    new_height, new_width = new_size

    new_y, new_x = np.meshgrid(np.linspace(0, old_height - 1, new_height), np.linspace(0, old_width - 1, new_width), indexing='ij')

    image_flat = img.flatten()

    A = np.column_stack([np.ones(old_height * old_width), np.repeat(np.arange(old_height), old_width), np.tile(np.arange(old_width), old_height)])

    coeffs, _, _, _ = np.linalg.lstsq(A, image_flat, rcond=None)
    new_coords = np.column_stack([np.ones(new_height * new_width), new_y.flatten(), new_x.flatten()])
    interpolated_image_flat = np.dot(new_coords, coeffs)
    interpolated_image = interpolated_image_flat.reshape(new_width, new_height)
    return interpolated_image


# im = Image.fromarray(nearest_neighbor(big_image_arr.copy(), (1000, 1000)))
# im.save("nn_big.jpeg")
# im = Image.fromarray(linear(big_image_arr.copy(), (1000, 1000)))
# im.save("linear_big.jpeg")
# im = Image.fromarray(polynomial_interpolation(big_image_arr.copy(), (1000, 1000)).astype(np.uint8))
# im.save("pi_big.jpeg")
# im = Image.fromarray(fourier_interpolation(big_image_arr.copy(), (1000, 1000)).astype(np.uint8))
# im.save("fi_big.jpeg")
# im = Image.fromarray(spline_interpolation(big_image_arr.copy(), (1000, 1000)).astype(np.uint8))
# im.save("si_big.jpeg")
# im = Image.fromarray(least_squares_interpolation(big_image_arr.copy(), (1000, 1000)).astype(np.uint8))
# im.save("lsi_big.jpeg")

# f, axarr = plt.subplots(1, 2)
# axarr[0].imshow(linear(big_image_arr.copy(), (1000, 1000)), cmap='gray')
# axarr[1].imshow(big_image_arr, cmap='gray')
# plt.show()
