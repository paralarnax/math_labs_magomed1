import numpy as np

def quaternion_mult(q1, q2):
  return np.array([
    q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
    q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
    q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
    q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
  ])

if __name__ == '__main__':
    q1 = np.array([0.5, 0.5, 0.5, 0.5])
    q2 = np.array([0.8, 0.2, 0.3, 0.4])

    result = quaternion_mult(q1, q2)
    
if __name__=="__main__":
    print(f"q1: {q1}")
    print(f"q2: {q2}")
    print(f"q1 * q2: {result}")
