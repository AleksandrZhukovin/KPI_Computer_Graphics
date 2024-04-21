import matplotlib.pyplot as plt
from random import randrange
from numpy.random import choice
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()


def leaf():
    a = [0.14, 0.43, 0.45, 0.49]
    b = [0.01, 0.52, -0.49, 0]
    c = [0, -0.45, 0.47, 0]
    d = [0.51, 0.5, 0.47, 0.51]
    e = [-0.08, 1.49, -1.62, 0.02]
    f = [-1.31, -0.75, -0.74, 1.62]
    x = [1]
    y = [1]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 4)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def spiral():
    a = [0.787879, -0.121212, 0.181818]
    b = [-0.424242, 0.257576, -0.136364]
    c = [0.242424, 0.151515, 0.090909]
    d = [0.859848, 0.053030, 0.181818]
    e = [1.758647, -6.721654, 6.086107]
    f = [1.408065, 1.377236, 1.568035]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = choice([0, 1, 2], 1, p=[0.9, 0.05, 0.05])[0]
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def mandelbrot_like():
    a = [0.202, 0.138]
    b = [-0.805, 0.665]
    c = [-0.689, -0.502]
    d = [-0.342, -0.222]
    e = [-0.373, 0.66]
    f = [-0.653, -0.277]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 2)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def tree():
    a = [0.0500, -0.0500, 0.0300, -0.0300, 0.5600, 0.1900, -0.3300]
    b = [0.0000, 0.0000, -0.1400, 0.1400, 0.4400, 0.0700, -0.3400]
    c = [0.0000, 0.0000, 0.0000, 0.0000, -0.3700, -0.1000, -0.3300]
    d = [0.4000, -0.4000, 0.2600, -0.2600, 0.5100, 0.1500, 0.3400]
    e = [-0.0600, -0.0600, -0.1600, -0.1600, 0.3000, -0.2000, -0.5400]
    f = [-0.4700, -0.4700, -0.0100, -0.0100, 0.1500, 0.2800, 0.3900]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 7)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def tree1():
    a = [0.0500, 0.0500, 0.6000, 0.5000, 0.5000, 0.5500]
    b = [0.6000, -0.5000, 0.5000, 0.4500, 0.5500, 0.4000]
    c = [0.0000, 0.0000, 0.6980, 0.3490, -0.5240, -0.6980]
    d = [0.0000, 0.0000, 0.6980, 0.3492, -0.5240, -0.6980]
    e = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    f = [0.0000, 1.0000, 0.6000, 1.1000, 1.0000, 0.7000]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 6)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def tree2():
    a = [0.0100, -0.0100, 0.4200, 0.4200]
    b = [0.0000, 0.0000, -0.4200, 0.4200]
    c = [0.0000, 0.0000, 0.4200, -0.4200]
    d = [0.4500, -0.4500, 0.4200, 0.4200]
    e = [0.0000, 0.0000, 0.0000, 0.0000]
    f = [0.0000, 0.4000, 0.4000, 0.4000]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 4)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def tree3():
    a = [0.1950, 0.4620, -0.6370, -0.0350, -0.0580]
    b = [-0.4880, 0.4140, 0.0000, 0.0700, -0.0700]
    c = [0.3440, -0.2520, 0.0000, -0.4690, 0.4530]
    d = [0.4430, 0.3610, 0.5010, 0.0220, -0.1110]
    e = [0.4431, 0.2511, 0.8562, 0.4884, 0.5976]
    f = [0.2452, 0.5692, 0.2512, 0.5069, 0.0969]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 5)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def leaf1():
    a = [0.0000, 0.7248, 0.1583, 0.3386]
    b = [0.2439, 0.0337, -0.1297, 0.3694]
    c = [0.0000, -0.0253, 0.3550, 0.2227]
    d = [0.3053, 0.7426, 0.3676, -0.0756]
    e = [0.0000, 0.2060, 0.1383, 0.0679]
    f = [0.0000, 0.2538, 0.1750, 0.0826]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 4)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def dollar():
    a = [0.38200, 0.11800, 0.11800, -0.30900, -0.30900, 0.38200]
    b = [0.00000, -0.36330, 0.36330, -0.22450, 0.22450, 0.00000]
    c = [0.00000, 0.36330, -0.36330, 0.22450, -0.22450, 0.00000]
    d = [0.38200, 0.11800, 0.11800, -0.30900, -0.30900, -0.38200]
    e = [0.30900, 0.36330, 0.51870, 0.60700, 0.70160, 0.30900]
    f = [0.57000, 0.33060, 0.69400, 0.30900, 0.53350, 0.67700]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 6)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def paporot():
    a = [0.0, 0.2, -0.15, 0.75]
    b = [0.0, -0.26, 0.28, 0.04]
    c = [0.0, 0.23, 0.26, -0.04]
    d = [0.16, 0.22, 0.24, 0.85]
    e = [0.0, 0.0, 0.0, 0.0]
    f = [0.0, 1.6, 0.44, 1.6]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = choice([0, 1, 2, 3], 1, p=[0.1, 0.08, 0.08, 0.74])[0]
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def another():
    a = [0.0, 0.2, -0.15, 0.85]
    b = [0.0, -0.26, 0.28, 0.04]
    c = [0.0, 0.23, 0.26, -0.04]
    d = [0.16, 0.22, 0.24, 0.85]
    e = [0.0, 0.0, 0.0, 0.0]
    f = [0.0, 1.6, 0.44, 1.6]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = choice([0, 1, 2, 3], 1, p=[0.01, 0.07, 0.07, 0.85])[0]
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def text():
    a = [0, 0.143, 0.143, 0, 0.119, -0.0123806, 0.0852291, 0.104432, -0.00814186, 0.093, 0, 0.119, 0.119, 0, 0.123998, 0, 0.071, 0, -0.121]
    b = [0.053, 0, 0, 0.53, 0, -0.0649723, 0.0506328, 0.00529117, -0.0417935, 0, 0.053, 0, 0, 0.053, -0.00183957, 0.053, 0, -0.053, 0]
    c = [-0.429, 0, 0, 0.429, 0, 0.423819, 0.420449, 0.0570516, 0.423922, 0, -0.429, 0, 0, 0.429, 0.000691208, 0.167, 0, -0.238, 0]
    d = [0, -0.053, 0.083, 0, 0.053, 0.00189797, 0.0156626, 0.0527352, 0.00415972, 0.053, 0, -0.053, 0.053, 0, 0.0629731, 0, 0.053, 0, 0.053]
    e = [-7.083, -5.619, -5.619, -3.952, -2.555, -1.226, -0.421, 0.976, 1.934, 0.861, 2.447, 3.363, 3.363, 3.972, 6.275, 5.215, 6.279, 6.805, 5.941]
    f = [5.43, 8.513, 2.057, 5.43, 4.536, 5.235, 4.569, 8.113, 5.37, 4.536, 5.43, 8.513, 1.487, 4.569, 7.716, 6.483, 5.298, 3.714, 1.487]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 19)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def dragon():
    a = [0.824074, 0.088272]
    b = [0.281428, 0.520988]
    c = [-0.212346, -0.463889]
    d = [0.864198, -0.377778]
    e = [-1.882290, 0.785360]
    f = [-0.110607, 8.095795]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = choice([0, 1], 1, p=[0.8, 0.2])[0]
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def branch():
    a = [0.387, 0.441, -0.468]
    b = [0.430, -0.091, 0.020]
    c = [0.430, -0.009, -0.113]
    d = [-0.387, -0.322, 0.015]
    e = [0.2560, 0.4219, 0.4]
    f = [0.5220, 0.5059, 0.4]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 3)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def tree_():
    a = [0, 0, 0.5]
    b = [-0.5, 0.5, 0]
    c = [0.5, -0.5, 0]
    d = [0, 0, 0.5]
    e = [0.5, 0.5, 0.25]
    f = [0, 0.5, 0.5]
    x = [0]
    y = [0]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 3)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def my1():
    a = [0.5, 0.5, -0.5]
    b = [0, 0.25, 0.75]
    c = [0, -0.25, 0]
    d = [0.5, 0.5, 0.25]
    e = [0, 0, 0.75]
    f = [0, 0, 0.5]
    x = [1]
    y = [1]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = choice([0, 1, 2], 1, p=[0.4, 0.3, 0.3])[0]
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def my2():
    a = [0.5, 0.5, 0.5]
    b = [0, 0, 0]
    c = [0, 0, 0]
    d = [0.5, 0.5, 0.25]
    e = [0, 0.5, 0.125]
    f = [0, 0, 0.216]
    x = [1]
    y = [1]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 3)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


def my3():
    a = [-0.5, 0.5]
    b = [-0.5, -0.5]
    c = [0.5, 0.5]
    d = [-0.5, 0.5]
    e = [4.9, 3.4]
    f = [1.2, -1.1]
    x = [1]
    y = [1]

    def anim(frame):
        for _ in range(1000):
            _ *= frame
            i = randrange(0, 2)
            x.append(x[_] * a[i] + b[i] * y[_] + e[i])
            y.append(c[i] * x[_] + d[i] * y[_] + f[i])
        ax.clear()
        ax.plot(x, y, '.', markersize=0.5)

    anim = FuncAnimation(fig, anim, frames=100, interval=1)
    plt.show()


my3()
