import matplotlib.pyplot as plt
import numpy as np

def get_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1 - cos_a
    x, y, z = axis

    # матрица поворота формула Родрига
    return np.array([
        [cos_a + x**2 * one_minus_cos,       x*y * one_minus_cos - z*sin_a, x*z * one_minus_cos + y*sin_a],
        [y*x * one_minus_cos + z*sin_a,     cos_a + y**2 * one_minus_cos,   y*z * one_minus_cos - x*sin_a],
        [z*x * one_minus_cos - y*sin_a,     z*y * one_minus_cos + x*sin_a, cos_a + z**2 * one_minus_cos]
    ])

def plot_rotation(axis, angle, vector):
    # расчет поворота
    rot_matrix = get_rotation_matrix(axis, angle)
    rotated_vector = np.dot(rot_matrix, vector)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    ax.plot([0, 1], [0, 0], [0, 0], 'r', label='X', alpha=0.5)
    ax.plot([0, 0], [0, 1], [0, 0], 'g', label='Y', alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, 1], 'b', label='Z', alpha=0.5)
    ax.scatter([0], [0], [0], c='k', marker='o') # Начало координат


    # исходный вектор 
    ax.plot([0, vector[0]], [0, vector[1]], [0, vector[2]], 
            'k--', linewidth=2, label='Исходный вектор')
    
    # повернутый вектор - мелкий пунктир
    ax.plot([0, rotated_vector[0]], [0, rotated_vector[1]], [0, rotated_vector[2]], 
            'k:', linewidth=2, label='Повернутый вектор')

    ax.text(vector[0], vector[1], vector[2], 
            f' ({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})', fontsize=9)
    ax.text(rotated_vector[0], rotated_vector[1], rotated_vector[2], 
            f' ({rotated_vector[0]:.2f}, {rotated_vector[1]:.2f}, {rotated_vector[2]:.2f})', fontsize=9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    limit = max(np.linalg.norm(vector), 1.0)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.legend()
    ax.set_title(f'Поворот на {np.degrees(angle):.1f}° вокруг оси {axis.round(2)}')
    plt.show()

if __name__ == '__main__':
    # Генерация случайных данных
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.uniform(-1, 1, 3)
    vector = np.array([0.5, 0.8, 0.2])

    plot_rotation(axis, angle, vector)