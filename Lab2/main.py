import numpy as np
import matplotlib.pyplot as plt

cube = [[1, 5, 5, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5],
        [1, 1, 5, 5, 1, 1, 1, 5, 5, 1, 5, 5, 5, 5, 1, 1],
        [1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 1, 1, 5, 5, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
cube = np.array(cube)

'''Scale'''
scale_matrix = np.zeros(shape=(4, 4))
for i in range(3):
    scale_matrix[i, i] = 2
scale_matrix[3, 3] = 1

move_matrix = np.zeros(shape=(4, 4))
for i in range(3):
    move_matrix[i, i] = 1
move_matrix[:, -1] = [-3, -3, -3, 1]

moved_cube = np.dot(move_matrix, cube)

scaled_cube_x2 = np.dot(scale_matrix, moved_cube)

scaled_cube_x2 = np.dot(abs(move_matrix), scaled_cube_x2)

for i in range(3):
    scale_matrix[i, i] = 0.5
scale_matrix[3, 3] = 1

moved_cube = np.dot(move_matrix, cube)

scaled_cube_x05 = np.dot(scale_matrix, moved_cube)

scaled_cube_x05 = np.dot(abs(move_matrix), scaled_cube_x05)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(cube[0, :].reshape((-1, 16)), cube[1, :].reshape((-1, 16)), cube[2, :].reshape((-1, 16)))
# ax.plot_wireframe(scaled_cube_x2[0, :].reshape((-1, 16)), scaled_cube_x2[1, :].reshape((-1, 16)),
#                   scaled_cube_x2[2, :].reshape((-1, 16)), color='green')
# ax.plot_wireframe(scaled_cube_x05[0, :].reshape((-1, 16)), scaled_cube_x05[1, :].reshape((-1, 16)),
#                   scaled_cube_x05[2, :].reshape((-1, 16)), color='red')
# ax.set_title('Scale')
# plt.show()

'''0 point|XY symetry'''
zero_symetry = cube * -1

XY_symetry = np.copy(cube)
XY_symetry[2, :] = XY_symetry[2, :] * -1

# ax.plot_wireframe(zero_symetry[0, :].reshape((-1, 16)), zero_symetry[1, :].reshape((-1, 16)),
#                   zero_symetry[2, :].reshape((-1, 16)), color='red')
# ax.plot_wireframe(XY_symetry[0, :].reshape((-1, 16)), XY_symetry[1, :].reshape((-1, 16)),
#                   XY_symetry[2, :].reshape((-1, 16)), color='yellow')
# ax.plot_wireframe(np.array([[7, -7, -7, 7, 7]]), np.array([[-7, -7, 7, 7, -7]]), np.array([[0, 0, 0, 0, 0]]),
#                   color='violet')
# ax.set_title('Symetry')
# plt.show()

'''Line turn'''
line = np.array([[-2, 1],
                 [0, 3],
                 [3, 6],
                 [1, 1]])
move_line_matrix = np.zeros((4, 4))
for i in range(4):
    move_line_matrix[i, i] = 1
move_line_matrix[:3, 3] = [2, 0, -3]

moved_cube = np.dot(move_line_matrix, cube)

turn_x = np.zeros((4, 4))
turn_x[0, 0], turn_x[3, 3] = 1, 1
turn_x[1:3, 1:3] = [[(1/np.sqrt(3))/np.sqrt(2/3), -(1/np.sqrt(3))/np.sqrt(2/3)],
                    [(1/np.sqrt(3))/np.sqrt(2/3), (1/np.sqrt(3))/np.sqrt(2/3)]]
turned_x_cube = np.dot(turn_x, moved_cube)

new_vector = np.dot(turn_x, np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 1]]).T)

turn_y = np.array([[new_vector[2, 0], 0, -new_vector[0, 0], 0],
                   [0, 1, 0, 0],
                   [new_vector[0, 0], 0, new_vector[2, 0], 0],
                   [0, 0, 0, 1]])
turned_y_cube = np.dot(turn_y, turned_x_cube)

turn_z = np.array([[0, -1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
turned_z_cube = np.dot(turn_z, turned_y_cube)

turn_y_again = np.array([[new_vector[2, 0], 0, new_vector[0, 0], 0],
                         [0, 1, 0, 0],
                         [-new_vector[0, 0], 0, new_vector[2, 0], 0],
                         [0, 0, 0, 1]])
turned_y_cube_a = np.dot(turn_y_again, turned_z_cube)

turn_x_again = np.zeros((4, 4))
turn_x_again[0, 0], turn_x_again[3, 3] = 1, 1
turn_x_again[1:3, 1:3] = [[(1/np.sqrt(3))/np.sqrt(2/3), (1/np.sqrt(3))/np.sqrt(2/3)],
                         [-(1/np.sqrt(3))/np.sqrt(2/3), (1/np.sqrt(3))/np.sqrt(2/3)]]
turned_x_cube_a = np.dot(turn_x_again, turned_y_cube_a)

for i in range(4):
    move_line_matrix[i, i] = 1
move_line_matrix[:3, 3] = [-2, 0, 3]
moved_cube_a = np.dot(move_line_matrix, turned_x_cube_a)

# ax.plot_wireframe(line[0, :].reshape((-1, 2)), line[1, :].reshape((-1, 2)), line[2, :].reshape((-1, 2)),
#                   color='violet')
# ax.plot_wireframe(moved_cube_a[0, :].reshape((-1, 16)), moved_cube_a[1, :].reshape((-1, 16)),
#                   moved_cube_a[2, :].reshape((-1, 16)), color='red')
# plt.show()

'''Square symetry'''
a = [3, 7, 9]
b = [5, -3, 4]
c = [9, 10, 3]

dots = np.array([[3, 5, 9, 3],
                 [7, -3, 10, 7],
                 [9, 4, 3, 9]])
A = np.linalg.det([[b[1]-a[1], b[2]-a[2]],
                   [c[1]-a[1], c[2]-a[2]]])

B = -np.linalg.det([[b[0]-a[0], b[2]-a[2]],
                    [c[0]-a[0], c[2]-a[2]]])

C = np.linalg.det([[b[0]-a[0], b[1]-a[1]],
                   [c[0]-a[0], c[1]-a[1]]])
D = -3*A - 7*B - 9*C

print(A, B, C, D)
M = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, D/C],
              [0, 0, 0, 1]])
M1 = np.array([[A/np.sqrt(A**2 + B**2), B/np.sqrt(A**2 + B**2), 0, 0],
               [-B/np.sqrt(A**2 + B**2), A/np.sqrt(A**2 + B**2), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
M2 = np.array([[C/np.sqrt(A**2 + B**2 + C**2), 0, -np.sqrt(A**2 + B**2)/np.sqrt(A**2 + B**2 + C**2), 0],
               [0, 1, 0, 0],
               [np.sqrt(A**2 + B**2)/np.sqrt(A**2 + B**2 + C**2), 0, C/np.sqrt(A**2 + B**2 + C**2), 0],
               [0, 0, 0, 1]])
M3 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
M4 = np.array([[C/np.sqrt(A**2 + B**2 + C**2), 0, np.sqrt(A**2 + B**2)/np.sqrt(A**2 + B**2 + C**2), 0],
               [0, 1, 0, 0],
               [-np.sqrt(A**2 + B**2)/np.sqrt(A**2 + B**2 + C**2), 0, C/np.sqrt(A**2 + B**2 + C**2), 0],
               [0, 0, 0, 1]])
M5 = np.array([[A/np.sqrt(A**2 + B**2), -B/np.sqrt(A**2 + B**2), 0, 0],
               [B/np.sqrt(A**2 + B**2), A/np.sqrt(A**2 + B**2), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
M6 = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, -D/C],
              [0, 0, 0, 1]])
cube_1 = np.dot(M, cube)
cube_2 = np.dot(M1, cube_1)
cube_3 = np.dot(M2, cube_2)
cube_4 = np.dot(M3, cube_3)
cube_5 = np.dot(M4, cube_4)
cube_6 = np.dot(M5, cube_5)
cube_7 = np.dot(M6, cube_6)
ax.plot_wireframe(dots[0, :].reshape((-1, 4)), dots[1, :].reshape((-1, 4)),
                  dots[2, :].reshape((-1, 4)))
# ax.plot_wireframe(cube_7[0, :].reshape((-1, 16)), cube_7[1, :].reshape((-1, 16)),
#                   cube_7[2, :].reshape((-1, 16)), color='red')
# plt.show()

'''Plane symetry again'''
# move_to_center = np.array([[1, 0, 0, -3],
#                            [0, 1, 0, -3],
#                            [0, 0, 1, -3],
#                            [0, 0, 0, 1]])
# moved_to_center = np.dot(move_to_center, cube)
#
# move_to_cross = np.array([[1, 0, 0, 0],
#                           [0, 1, 0, 0],
#                           [0, 0, 1, -D/C],
#                           [0, 0, 0, 1]])
# moved_to_cross = np.dot(move_to_cross, moved_to_center)
#
# turn_y = np.array([[(-D/B)/np.sqrt((D/A)**2+(D/B)**2), 0, -(-D/A)/np.sqrt((D/A)**2+(D/B)**2), 0],
#                    [0, 1, 0, 0],
#                    [(-D/A)/np.sqrt((D/A)**2+(D/B)**2), 0, (-D/B)/np.sqrt((D/A)**2+(D/B)**2), 0],
#                    [0, 0, 0, 1]])
#
# turn_z = np.array([[np.sqrt((D/C)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), -np.sqrt((D/A)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), 0, 0],
#                    [np.sqrt((D/A)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), np.sqrt((D/C)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), 0, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 1]])
#
# turn_x = np.array([[1, 0, 0, 0],
#                    [0, np.sqrt((D/A)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), -np.sqrt((D/C)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), 0],
#                    [0, np.sqrt((D/C)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), np.sqrt((D/A)**2 + (D/B)**2)/np.sqrt((D/A)**2+(D/B)**2), 0],
#                    [0, 0, 0, 1]])
# R = np.dot(np.dot(turn_z, turn_x), turn_y)
#
# new = np.dot(R, moved_to_cross)
#
# new = np.dot(np.array([[1, 0, 0, 0],
#                           [0, 1, 0, 0],
#                           [0, 0, 1, D/C],
#                           [0, 0, 0, 1]]), new)
# new = np.dot(np.array([[1, 0, 0, 3],
#                            [0, 1, 0, 3],
#                            [0, 0, 1, 3],
#                            [0, 0, 0, 1]]), new)

new_cube = None

projection = None

for x, y, z, o in cube.T:
    coef = (-D-A*x-B*y-C*z)/(A**2 + B**2 + C**2)
    norm_point = np.array([[A*coef + x, B*coef + y, C*coef + z]]).T
    sym_point = norm_point*2 - np.array([[x, y, z]]).T
    if new_cube is None:
        new_cube = np.copy(sym_point)
    else:
        new_cube = np.hstack((new_cube, sym_point))

    if projection is None:
        projection = np.copy(norm_point)
    else:
        projection = np.hstack((projection, norm_point))


# ax.plot_wireframe(new[0, :].reshape((-1, 16)), new[1, :].reshape((-1, 16)),
#                   new[2, :].reshape((-1, 16)), color='red')
# ax.plot_wireframe(projection[0, :].reshape((-1, 16)), projection[1, :].reshape((-1, 16)),
#                   projection[2, :].reshape((-1, 16)), color='green')
# ax.plot_wireframe(new_cube[0, :].reshape((-1, 16)), new_cube[1, :].reshape((-1, 16)),
#                   new_cube[2, :].reshape((-1, 16)), color='yellow')
plt.show()