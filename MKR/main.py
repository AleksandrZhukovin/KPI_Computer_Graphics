"""
Варіант 6
Визначити фрактальну розмірність фрактала на площині, що складається з
точок (x; y), де x ∈ [0; 1], y ∈ [0; 1], причому в десятковому представленні
чисел x і y відсутні цифри 3 та 7. Розробити програмне забезпечення для
побудови даного фрактала.
"""

import numpy as np
import matplotlib.pyplot as plt


def fractal_build():
    # отримуємо необхідний діапазон чисел
    nums_range = []
    for i in np.arange(0, 1, 0.01):
        if '3' in str(i) or '7' in str(i):
            continue
        nums_range.append(i)

    # створюємо матрицю точок
    points = np.array([0, 0])
    for x_i in nums_range:
        for y_i in nums_range:
            points = np.vstack((points, np.array([x_i, y_i])))

    plt.plot(points[:, 0], points[:, 1], '.')
    plt.show()


fractal_build()
