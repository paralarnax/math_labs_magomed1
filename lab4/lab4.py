import numpy as np
from sympy import Symbol, nsolve
import sympy as sp


def get_matrix(link_length_a, link_twist_alpha, link_offset_d, joint_angle_theta):
    try:
        a = float(link_length_a)
        alpha = float(link_twist_alpha)
        d = float(link_offset_d)
        theta = float(joint_angle_theta)

        T = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

        T = np.round(T, 3)

        return T

    except (TypeError, ValueError):
        print("Ne chisla")
        return None

def RPR_FK(O1, d2, O3):
    a = 10
    b = 5

    T0 = get_matrix(0, 0, 0, 0)
    T01 = get_matrix(0, -3 * np.pi / 4, a, O1)
    T12 = get_matrix(0, -np.pi / 2, d2, -np.pi / 2)
    T23 = get_matrix(0, np.pi / 2, 0, O3 + np.pi / 4)
    T34 = get_matrix(0, 0, b, np.pi / 2)

    T = [T0, T01, T12, T23, T34]

    points = list()

    zero_point = np.array([0, 0, 0, 1])
    for i in range(1, len(T)+1):
        mat = np.eye(4)

        for j in range(i):
            mat = np.matmul(mat, T[j])

        if i == len(T):
            R03 = mat[0:3, 0:3]

        pos = np.matmul(mat, zero_point)
        points.append(list(np.around(pos, 3))[:-1])

    points = np.round(points, 3)
    R03 = np.round(R03, 8)
    points = np.delete(points, 0, 0)
    return points, R03


def RPR_IK(x, y, z, R03):
    a = 10
    b = 5

    O1 = Symbol('O1')
    d2 = Symbol('d2')
    O3 = Symbol('O3')

    '''
    определение матрицы поворота 
    R = [cos(θ) - sin(θ)n_x*n_y - sin(θ)n_x*n_z  sin(θ)n_y*n_x + cos(θ) - sin(θ)n_y*n_z  sin(θ)n_z*n_x + sin(θ)n_y*n_z  cos(θ)]

    R03[0, 0] = cos(O1)
    R03[1, 0] = sin(O1)
    R03[2, 0] = 0
   
    R03[0, 1] = -sin(O1) * cos(O3) + cos(O1) * sin(O3) * sin(np.pi/4)
    R03[1, 1] = cos(O1) * cos(O3) + sin(O1) * sin(O3) * sin(np.pi/4)
    R03[2, 1] = sin(O3) * cos(np.pi/4) + cos(O3) * sin(np.pi/4)
   
    R03[0, 2] = sin(O1) * sin(O3) + cos(O1) * cos(O3) * sin(np.pi/4)
    R03[1, 2] = -cos(O1) * sin(O3) + sin(O1) * cos(O3) * sin(np.pi/4)
    R03[2, 2] = cos(O3) * cos(np.pi/4) - sin(O3) * sin(np.pi/4)
    '''

    f1 = sp.cos(O1) - R03[0, 0]
    f2 = sp.sin(O1) - R03[1, 0]
    f3 = R03[2, 0]

    f7 = R03[0, 1] - sp.sin(np.pi/4)*sp.cos(np.pi/4+O3)*sp.sin(O1) + sp.cos(np.pi/4)*sp.sin(np.pi/4+O3)*sp.sin(O1)
    f8 = R03[1, 1] - sp.cos(np.pi/4)*sp.sin(np.pi/4+O3)*sp.cos(O1) + sp.sin(np.pi/4)*sp.cos(np.pi/4+O3)*sp.cos(O1)
    f9 = R03[2, 1] + sp.sin(np.pi/4)*sp.cos(np.pi/4+O3) + sp.cos(np.pi/4)*sp.sin(np.pi/4+O3)

    f10 = R03[0, 2] + sp.sin(np.pi/4)*sp.cos(np.pi/4+O3)*sp.sin(O1) + sp.cos(np.pi/4)*sp.sin(np.pi/4+O3)*sp.sin(O1)
    f11 = R03[1, 2] - sp.sin(np.pi/4)*sp.cos(np.pi/4+O3)*sp.cos(O1) - sp.cos(np.pi/4)*sp.sin(np.pi/4+O3)*sp.cos(O1)
    f12 = R03[2, 2] - sp.cos(np.pi/4)*sp.sin(np.pi/4+O3) + sp.sin(np.pi/4)*sp.cos(np.pi/4+O3)

    f4 = z - a + d2 * sp.cos(np.pi / 4) - b * sp.sin(O3) #z
    f5 = y - (d2 * sp.sin(np.pi / 4) + b * sp.cos(O3)) * sp.cos(O1) #y
    f6 = x + (d2 * sp.sin(np.pi / 4) + b * sp.cos(O3)) * sp.sin(O1) #x

    equations = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

    try:
        result = nsolve(equations, (O1, d2, O3), (np.pi, np.pi, np.pi), dict=True, prec=2, maxsteps=100000)[0]


        return [round(float(result[O1]), 3), round(float(result[d2]),3), round(float(result[O3]),3)]
    except Exception as e:
        print(e)
        return []


if __name__=="__main__":
    print("task1 part1. Zero pos: 0 5 0")
    pos, R = RPR_FK(0, 5, 0)
    print("pos:\n", pos, "\n R:\n", R)

    print("task1 part2. Not zero pos: pi -10 -pi/2")
    pos, R = RPR_FK(np.pi, -10, np.pi/2)
    print("pos:\n", pos, "\n R:\n", R)

    print("task2 part1")
    ik_sol = RPR_IK(*pos[-1], R)

    if not ik_sol:
        exit('Incorrect input for RPR_IK')

    pos_ik, R_ik = RPR_FK(*ik_sol)
    print('pos_ik\n', pos_ik[-1])
    print('pos\n', pos[-1])







