import numpy as np
from pyquaternion import Quaternion

# Ручная функция перемножения кватернионов
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


if __name__ == '__main__':
    # Задаём кватернионы как массивы numpy и перемножаем их через ранее реализованную функцию
    q1 = np.array([1, 0, 1, 0])
    q2 = np.array([1, 0.5, 0.5, 0.75])
    result = multiply_quaternions(q1, q2)

    # Создаём и перемножаем кватернионы с идентичными значениями через библиотеку
    q1_ch = Quaternion(w=1, x=0, y=1, z=0)
    q2_ch = Quaternion(w=1, x=0.5, y=0.5, z=0.75)
    result_ch = q1_ch * q2_ch

if __name__=="__main__":
    print(f"Первый кватернион: {q1}")
    print(f"Второй кватернион: {q2}")
    print(f"Умножение кватернионов: {result}")
    print(f"Умножение кватернионов через библиотечную функцию: {result_ch}")
