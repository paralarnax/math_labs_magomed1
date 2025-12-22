import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix(axis, angle):
  axis = axis / np.linalg.norm(axis) #нормализация оси вращения
  a = np.cos(angle)
  b = np.sin(angle)
  v = axis
  return np.array([
    [a + v[0]*2*(1-a), v[0]*v[1]*(1-a) - v[2]*b, v[0]*v[2]*(1-a) + v[1]*b],
    [v[1]*v[0]*(1-a) + v[2]*b, a + v[1]*2*(1-a), v[1]*v[2]*(1-a) - v[0]*b],
    [v[2]*v[0]*(1-a) - v[1]*b, v[2]*v[1]*(1-a) + v[0]*b, a + v[2]*2*(1-a)]
  ])

def plot_rotation(axis, angle, vector):
  rot_matrix = rotation_matrix(axis, angle)
  rotated_vector = np.dot(rot_matrix, vector)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.plot([0, 1], [0, 0], [0, 0], 'r', label='X')
  ax.plot([0, 0], [0, 1], [0, 0], 'g', label='Y')
  ax.plot([0, 0], [0, 0], [0, 1], 'b', label='Z')
  ax.scatter([0], [0], [0], c='k', marker='o')

  ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], 'k--', label='Исходный вектор')
  ax.plot([0, rotated_vector[0]], [0, rotated_vector[1]], [0, rotated_vector[2]], 'k:', label='Повернутый вектор')

  ax.text(1.1, 0, 0, 'X', color='r', fontsize=12)
  ax.text(0, 1.1, 0, 'Y', color='g', fontsize=12)
  ax.text(0, 0, 1.1, 'Z', color='b', fontsize=12)
  ax.text(vector[0], vector[1], vector[2], f'({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})', color='k', fontsize=10)
  ax.text(rotated_vector[0], rotated_vector[1], rotated_vector[2], f'({rotated_vector[0]:.2f}, {rotated_vector[1]:.2f}, {rotated_vector[2]:.2f})', color='k', fontsize=10)


  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  ax.set_zlim([-1, 1])
  ax.legend()
  ax.set_title('Поворот вектора в 3D пространстве')
  plt.show()

angle = np.random.rand() * 2 * np.pi
axis = np.random.rand(3)
vector = np.random.rand(3)

print("Угол: ", angle)
print("Ось: ", axis)
print("Вектор: ", vector)


plot_rotation(axis, angle, vector)