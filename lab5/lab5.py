import numpy as np 
 
# матрица преобразований для одного звена робота 
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
 
 
def task_1(O1, O2, O3, O4, O5, v_O1, v_O2, v_O3, v_O4, v_O5, debug=False): 
    a = 3 
    b = 5.75 
    c = 7.375 
    d = 4.125 
    e = 1.125 
    # кгловые скорости
    v_O = np.array([[v_O1, v_O2, v_O3, v_O4, v_O5]])
    if debug:
        print(v_O)
 
    T1 = get_matrix(0, -np.pi / 2, a, O1) 
    T12 = get_matrix(b, 0, 0, O2 - np.pi / 2) 
    T23 = get_matrix(c, 0, 0, O3 + np.pi / 2) 
    T34 = get_matrix(0, -np.pi / 2, 0, O4 - np.pi / 2) 
    T45 = get_matrix(0, 0, d, O5) 
 
    T01 = T1 
    T02 = np.matmul(T01, T12) 
    T03 = np.matmul(T02, T23) 
    T04 = np.matmul(T03, T34) 
    T05 = np.matmul(T04, T45) 
 
    T_matrices = [T01, T02, T03, T04, T05]

    # базовая система координат плюс направления каждого звена в глобальной системе координат
    Z = np.array([np.array([0, 0, 1])] + [T[0:3, 2] for T in T_matrices])
    #print(Z)
    
    # координаты точек каждого звена в глобальной системе координат
    P = np.stack([T0i[0:3, 3] for i, T0i in enumerate([T01, T02, T03, T04, T05])], axis=0)
    #print(P)
    
    # Добавляем в начало P строку 0 0 0 , т.к это базовая точка
    P = np.vstack((np.array([0, 0, 0]), P))
    #print(P)
    
    # якобиан линейно   скорости (гометрический подход через век. проивзедения)
    J_l = np.array([np.cross(Z[i], P[-1]-P[i]) for i in range(5)]) # вклад каждого звена в линейную скорость
    #print(J_l)
    
    # скобиан угловой скорости 
    J_a = Z[:5]
    J_l = J_l.T
    J_a = J_a.T

    # размерность (n, 3) и (1, n) 
    v = J_l.dot(v_O.T) #  вектор линейно скорости конечно эффектора
    w = J_a.dot(v_O.T) # вектор угловой скорости конечного эффектора
    return v, w 
 
 
def task_2(O1, O2, O3, O4, O5, O6, v_O1, v_O2, v_O3, v_O4, v_O5, v_O6, debug=False): 
 
    a = 13.0 
    b = 2.5 
    c = 8.0 
    d = 2.5 
    e = 8.0 
    f = 2.5 
 
    v_O = np.array([[v_O1, v_O2, v_O3, v_O4, v_O5, v_O6]]) 
    if debug:
        print(v_O)
 
    T1 = get_matrix(0, np.pi / 2, a, O1) 
    T12 = get_matrix(c, 0, -b, O2) 
    T23 = get_matrix(0, -np.pi / 2, -d, O3) 
    T34 = get_matrix(0, np.pi / 2, e, O4) 
    T45 = get_matrix(0, -np.pi / 2, 0, O5) 
    T56 = get_matrix(0, 0, f, O6) 
 
    T01 = T1 
    T02 = np.matmul(T01, T12)
    T03 = np.matmul(T02, T23)
    T04 = np.matmul(T03, T34)
    T05 = np.matmul(T04, T45)
    T06 = np.matmul(T05, T56)
 
    T_matrices = [T01, T02, T03, T04, T05, T06]
    Z = np.array([np.array([0, 0, 1])] + [T[0:3, 2] for T in T_matrices])
    P = np.stack([T0i[0:3, 3] for i, T0i in enumerate([T01, T02, T03, T04, T05, T06])], axis=0)
    P = np.vstack((np.array([0, 0, 0]), P))

    J_l = np.array([np.cross(Z[i], P[-1]-P[i]) for i in range(6)])
    J_a = Z[:6]

    J_l = J_l.T
    J_a = J_a.T

    v = J_l.dot(v_O.T)
    w = J_a.dot(v_O.T)
 
    return v, w 
 
 
if __name__ == '__main__': 
    O1 = np.pi/2 
    O2 = -np.pi/2 
    O3 = np.pi/2 
    O4 = np.pi/3 
    O5 = np.pi/2 
 
    v_O1 = 0.1 
    v_O2 = 0.3 
    v_O3 = 0.2 
    v_O4 = -0.1 
    v_O5 = 0.6 

    print("\nTask1\ncalculated joint velocity")
    linear_velocity, angular_velocity = task_1(O1, O2, O3, O4, O5, v_O1, v_O2, v_O3, v_O4, v_O5) 
    print("linear:")
    print(f"X: {linear_velocity[0,0]:.3f}")
    print(f"Y: {linear_velocity[1,0]:.3f}")
    print(f"Z: {linear_velocity[2,0]:.3f}")

    print("angular:")
    print(f"X: {angular_velocity[0,0]:.3f}")
    print(f"Y: {angular_velocity[1,0]:.3f}")
    print(f"Z: {angular_velocity[2,0]:.3f}")
 
    O1 = np.pi/2 
    O2 = -np.pi/2 
    O3 = np.pi/4 
    O4 = -np.pi/6 
    O5 = np.pi/8 
    O6 = -np.pi/3 
 
    v_O1 = 0.1 
    v_O2 = -0.2 
    v_O3 = 0.3 
    v_O4 = 0.1 
    v_O5 = 0.4 
    v_O6 = -0.6 
 
    print("\nTask2\ncalculated joint velocity")
    linear_velocity, angular_velocity = task_2(O1, O2, O3, O4, O5, O6, v_O1, v_O2, v_O3, v_O4, v_O5, v_O6) 
    print("linear:")
    print(f"X: {linear_velocity[0,0]:.3f}")
    print(f"Y: {linear_velocity[1,0]:.3f}")
    print(f"Z: {linear_velocity[2,0]:.3f}")

    print("angular:")
    print(f"X: {angular_velocity[0,0]:.3f}")
    print(f"Y: {angular_velocity[1,0]:.3f}")
    print(f"Z: {angular_velocity[2,0]:.3f}")