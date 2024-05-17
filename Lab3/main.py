from turtle import Turtle, Screen, tracer, done, update
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


tracer(False)
t = Turtle()
screen = Screen()


def fractal1(length=10, angle=90):  # Хрестоподібний фрактал
    x = []
    y = []
    axiom = 'f+xf+f+xf'
    x_rep = 'xf-f+f-xf+f+xf-f+f-x'
    for i in range(3):
        axiom = axiom.replace('x', x_rep)
    for action in axiom:
        if action == 'f':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal2(length=5, angle=90):  # Квадратна сніжинка
    axiom = 'F'
    f_rep = 'F-F+F+F-F'
    x = []
    y = []
    for i in range(4):
        axiom = axiom.replace('F', f_rep)
    t.up()
    t.goto(-300, 300)
    t.down()
    for action in axiom:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal3(length=5, angle=90):  # Квадратний острівець Коха
    axiom = 'F+F+F+F'
    f_rep = 'F+F-F-FFF+F+F-F'
    x = []
    y = []
    for i in range(3):
        axiom = axiom.replace('F', f_rep)
    t.up()
    t.goto(-300, 0)
    t.down()
    for action in axiom:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal4(length=5, angle=90):  # Фрактальна дошка
    axiom = 'F+F+F+F'
    f_rep = 'FF+F+F+F+FF'
    x = []
    y = []
    for i in range(4):
        axiom = axiom.replace('F', f_rep)
    t.up()
    t.goto(-350, -300)
    t.down()
    for action in axiom:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal5(length=10, angle=60):  # Наконечник стріли Серпінського
    axiom = 'YF'
    x_rep = 'YF+XF+Y'
    y_rep = 'XF-YF-X'
    alg = axiom
    x = []
    y = []
    for i in range(5):
        ind = 0
        for s in axiom:
            if s == 'X':
                s = x_rep
            elif s == 'Y':
                s = y_rep
            else:
                s = s
            alg = alg[:ind] + s + alg[ind+1:]
            ind += len(s)
        axiom = alg
    t.up()
    t.goto(-200, 200)
    t.down()
    for action in alg:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal6(length=5, angle=36):  # П’ятикутна фрактальна сніжинка
    axiom = 'F++F++F++F++F'
    f_rep = 'F++F++F+++++F-F++F'
    x = []
    y = []
    for i in range(4):
        axiom = axiom.replace('F', f_rep)
    t.up()
    t.goto(-200, -100)
    t.down()
    for action in axiom:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal7(length=2, angle=90):  # Крива дракона
    axiom = 'FX'
    x_rep = 'X+YF+'
    y_rep = '-FX-Y'
    x = []
    y = []
    alg = axiom
    for i in range(15):
        ind = 0
        for s in axiom:
            if s == 'X':
                s = x_rep
            elif s == 'Y':
                s = y_rep
            else:
                s = s
            alg = alg[:ind] + s + alg[ind+1:]
            ind += len(s)
        axiom = alg
    t.up()
    t.goto(-100, 100)
    t.down()
    for action in alg:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y


def fractal8(length=2, angle=45):  # Крива Леві
    axiom = 'F'
    f_rep = '-F++F-'
    x = []
    y = []
    for i in range(15):
        axiom = axiom.replace('F', f_rep)
    t.up()
    t.goto(-170, 200)
    t.down()
    for action in axiom:
        if action == 'F':
            t.forward(length)
        elif action == '+':
            t.left(angle)
        elif action == '-':
            t.right(angle)
        x.append(t.pos()[0])
        y.append(t.pos()[1])
    return x, y

# fractal2()

xs = []
ys = []

for a in range(0, 90, 5):
    res = fractal7(angle=a)
    xs.append(res[0])
    ys.append(res[1])

fig, ax = plt.subplots()


def anim(frame):
    ax.clear()
    ax.plot(xs[frame], ys[frame])


ani = FuncAnimation(fig, anim, frames=len(xs), interval=150)

# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('Крива Дракона.gif')

plt.show()
update()
done()
