import matplotlib.pyplot as plt


def octagon():
    x = [3, 6, 9, 12, 12, 9, 6, 3, 3]
    y = [6, 3, 3, 6, 9, 12, 12, 9, 6]

    mid_x = (12+3)/2
    mid_y = (9+6)/2

    cen_x = [i - mid_x for i in x]
    cen_y = [i - mid_y for i in y]
    scaled_x = [i/1.5 + mid_x for i in cen_x]
    scaled_y = [i/1.5 + mid_y for i in cen_y]

    sim_x = [-i for i in scaled_x]
    sim_y = [-i for i in scaled_y]

    line_x = [-2, 6]
    line_y = [1, -6]
    line_center_x = (-2 + 1) / 2
    line_center_y = 0

    line_sim_x = [-(i - line_center_x) + line_center_x for i in x]
    line_sim_y = [-(i - line_center_y) + line_center_y for i in y]

    plt.plot(x, y, label='Start figure')
    plt.plot(cen_x, cen_y)
    plt.plot(scaled_x, scaled_y, label='Scaled figure')
    plt.plot([-2, 1], [6, -6], label='Line')
    plt.plot(sim_x, sim_y, label='Symmetry about the 0 point')
    plt.plot(line_sim_x, line_sim_y, label='Symmetry about a line')
    # plt.scatter((12+3)/2, (9+6)/2)
    plt.grid()
    plt.legend()
    plt.show()


def variant_2():
    with open('img.txt', 'r') as file:
        x = [float(i.split()[0]) for i in file.readlines()]

    with open('img.txt', 'r') as file:
        y = [float(i.split()[1]) for i in file.readlines()]

    mid_x = (max(x) + min(x)) / 2
    mid_y = (max(y) + min(y)) / 2

    scaled_x = [(i - mid_x) / 1.5 + mid_x for i in x]
    scaled_y = [(i - mid_y) / 1.5 + mid_y for i in y]

    sim_x = [-i for i in scaled_x]
    sim_y = [-i for i in scaled_y]

    line_center_x = (20 + 35) / 2
    line_center_y = 10

    line_sim_x = [-(i - line_center_x) + line_center_x for i in x]
    line_sim_y = [-(i - line_center_y) + line_center_y for i in y]

    plt.plot(x, y, label='Base figure')
    plt.plot(scaled_x, scaled_y, label='Scaled figure')
    plt.plot(sim_x, sim_y, label='Symmetry about the 0 point')
    plt.plot([20, 35], [50, -30], label='Line')
    plt.plot(line_sim_x, line_sim_y, label='Symmetry about a line')
    plt.grid()
    plt.legend()
    plt.show()


# octagon()
variant_2()
