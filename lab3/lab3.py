import numpy as np

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


def get_tool_pose(O1, O2, O3, O4, O5, g):
    a = 3
    b = 5.75
    c = 7.375
    d = 4.125
    e = 1.125

    T0 =  get_matrix(0, 0, 0, 0)
    T01 = get_matrix(0, -np.pi / 2, a, O1)
    T12 = get_matrix(b, 0, 0, O2 - np.pi / 2)
    T23 = get_matrix(c, 0, 0, O3 + np.pi / 2)
    T34 = get_matrix(0, -np.pi / 2, 0, O4 - np.pi / 2)
    T45 = get_matrix(0, 0, d, O5)

    T = [T0, T01, T12, T23, T34, T45]

    result = []

    zero_point = np.array([0, 0, 0, 1])
    for i in range(1, len(T)):
        mat = np.eye(4)

        for j in range(i):
            mat = np.matmul(mat, T[j])

        pos = np.matmul(mat, zero_point)
        result.append(list(np.around(pos, 3))[:-1])

    tool_points = np.array([[0, 0, -e, 1],
                            [g/2, 0, -e, 1],
                            [-g/2, 0, -e, 1],
                            [g/2, 0, 0, 1],
                            [-g/2, 0, 0, 1]], float)

    T05 = np.eye(4)
    for mat in T:
        T05 = np.matmul(T05, mat)

    for point in tool_points:
        pos = np.matmul(T05, point)
        result.append(list(np.around(pos, 3))[:-1])

    result = np.round(result, 3)
    return result

if __name__ == '__main__':
    print("task1")
    print(get_matrix(5, 0, 3, np.pi/2))

    answer_2_1 = get_tool_pose(0, 0, 0, 0, 0, 0)

    print("\ntask2 part1")
    for item in answer_2_1:
        print(item)

    print("\ntask2 part2")
    answer_2_2 = get_tool_pose(np.pi, np.pi/2, np.pi/2, -np.pi/2, -np.pi/6, 2)
    for item in answer_2_2:
        print(item)
