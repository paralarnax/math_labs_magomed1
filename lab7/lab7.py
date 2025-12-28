import matplotlib.pyplot as plt
import random
import keyboard
from time import sleep

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Triangle:
    def __init__(self, a=Point(), b=Point(), c=Point()):
        self.ds = [a, b, c]

    def plot(self, color='b'):
        x = [p.x for p in self.ds] + [self.ds[0].x]
        y = [p.y for p in self.ds] + [self.ds[0].y]
        plt.plot(x, y, color + '-')
        plt.fill(x,y, color='gray', alpha=0.2)

def cross_product(o, a, b):
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

def is_point_in_triangle(p, t):
    d1 = cross_product(t.ds[0], t.ds[1], p)
    d2 = cross_product(t.ds[1], t.ds[2], p)
    d3 = cross_product(t.ds[2], t.ds[0], p)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def do_intersect(a1, a2, b1, b2):
    d = cross_product(a1, a2, b1)
    e = cross_product(a1, a2, b2)
    f = cross_product(b1, b2, a1)
    g = cross_product(b1, b2, a2)
    return (d > 0 and e < 0 or d < 0 and e > 0) and (f > 0 and g < 0 or f < 0 and g > 0)


def is_intersect_triangle(a, b):
    if is_point_in_triangle(a.ds[0], b) or is_point_in_triangle(a.ds[1], b) or is_point_in_triangle(a.ds[2], b) or \
       is_point_in_triangle(b.ds[0], a) or is_point_in_triangle(b.ds[1], a) or is_point_in_triangle(b.ds[2], a):
        return True
    for i in range(3):
        for j in range(3):
            if do_intersect(a.ds[i], a.ds[(i + 1) % 3], b.ds[j], b.ds[(j + 1) % 3]):
                return True
    return False



def main():
    a = Triangle(
            Point(90, 50), 
            Point(150, round(random.uniform(0, 300), 1)), 
            Point(round(random.uniform(0, 300), 1), 100)
        )
    b= Triangle(
            Point(50, 50), 
            Point(150, round(random.uniform(0, 300), 1)), 
            Point(round(random.uniform(0, 300), 1), 100)
        )
    
    if is_intersect_triangle(a, b):
        print("Yes.")
    else:
        print("No!")
    a.plot('r')
    b.plot('g')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Triangles intersect is {is_intersect_triangle(a, b)}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 300)
    plt.ylim(0, 300)
    plt.grid(True)
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')
    plt.clf()

if __name__=="__main__":
    i = 0
    while i < 10:
        i = i + 1
        main()
        continue