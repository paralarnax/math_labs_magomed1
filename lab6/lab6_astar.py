import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    return path[::-1]

def aStar_grid(input_map, start_coords, dest_coords, draw_map=True):
    nrows, ncols = input_map.shape
    map = input_map  
    g_score = np.full(map.shape, np.inf) # расстояние от стартовой до текущей
    h_score = np.full(map.shape, np.inf) # эвричтиская оценка 
    f_score = np.full(map.shape, np.inf) # сумма
    parent = {}

    g_score[start_coords] = 0
    h_score[start_coords] = heuristic(start_coords, dest_coords)
    f_score[start_coords] = g_score[start_coords] + h_score[start_coords]

    priority_queue = [(f_score[start_coords], start_coords)] # нужно обработать
    expanded_nodes = set() # обработанные клетки
    frames = [] # карта нка ждом шагу

    while priority_queue:
        f, current = heapq.heappop(priority_queue)
        if current == dest_coords:
            path = reconstruct_path(parent, current)
            return path, expanded_nodes, frames
        if current in expanded_nodes:
            continue
        expanded_nodes.add(current)
        map[current] = 2

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < nrows and 0 <= neighbor[1] < ncols and map[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score[neighbor] = heuristic(neighbor, dest_coords)
                    f_score[neighbor] = tentative_g_score + h_score[neighbor]
                    heapq.heappush(priority_queue, (f_score[neighbor], neighbor))

        if draw_map:
            frames.append(np.copy(map))

    return None, expanded_nodes, frames

def visualize_path(input_map, path, frames):
    if path is None:
        print("Путь не найден")
        return

    fig, ax = plt.subplots()
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'red', 'green'])

    def animate(i):
        ax.clear()
        ax.imshow(frames[i], cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('Поиск А*')

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=5)

    # Final path visualization
    input_map_with_path = np.copy(input_map)
    for cell in path:
        input_map_with_path[cell] = 3

    input_map_with_path[path[0]] = 4
    input_map_with_path[path[-1]] = 4

    plt.figure()
    plt.imshow(input_map_with_path, cmap=cmap, interpolation='nearest')
    plt.title('Поиск A* кратчайший путь')
    plt.show()
    plt.show()

def generate_random_obstacles(rows, cols, obstacle_probability):
    return np.random.choice([0, 1], size=(rows, cols), p=[1 - obstacle_probability, obstacle_probability])


if __name__=="__main__":
    obstacle_probability = 0.2
    start_cell = (0, 0)

    def stable_micro_input_map_task():
        micro_end_cell = (4, 4)
        stable_micro_input_map = np.array([
            [0, 0, 0, 0, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
        print(stable_micro_input_map)
        route, num_expanded, frames = aStar_grid(stable_micro_input_map, start_cell, micro_end_cell, draw_map=True)
        visualize_path(stable_micro_input_map, route, frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    def micro_input_map_task():
        micro_end_cell = (4, 4)
        micro_input_map = generate_random_obstacles(5, 5, obstacle_probability)
        print(micro_input_map)
        route, num_expanded, frames = aStar_grid(micro_input_map, start_cell, micro_end_cell, draw_map=True)
        visualize_path(micro_input_map, route, frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()


    def normal_input_cell_task():
        normal_end_cell = (8, 9)
        normal_input_map = generate_random_obstacles(10, 10, obstacle_probability)
        print(normal_input_map)
        route, num_expanded, frames = aStar_grid(normal_input_map, start_cell, normal_end_cell, draw_map=True)
        visualize_path(normal_input_map, route, frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    def big_input_cell_task():
        big_end_cell=(13, 14)
        big_input_map = generate_random_obstacles(15, 15, obstacle_probability)
        print(big_input_map)
        route, num_expanded,frames = aStar_grid(big_input_map, start_cell, big_end_cell, draw_map=True)
        visualize_path(big_input_map, route,frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    def huge_task():
        big_end_cell=(199, 199)
        big_input_map = generate_random_obstacles(200, 200, obstacle_probability)
        print(big_input_map)
        route, num_expanded,frames = aStar_grid(big_input_map, start_cell, big_end_cell, draw_map=True)
        visualize_path(big_input_map, route,frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    
    def stable_normal_map_task():
        '''
        карты из лабораторной работы
        '''
        test_start_cell = (6, 2)
        test_dest_cell  = (8, 9)
        stable_test_input_map = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        route, num_expanded,frames = aStar_grid(stable_test_input_map, test_start_cell, test_dest_cell, draw_map=True)
        print(route)
        visualize_path(stable_test_input_map, route,frames)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()


    execution_time = timeit.timeit(stable_normal_map_task, number=1)
    print(f"Время выполнения stable_normal_map_task: {execution_time:.4f} секунд")

    

    
    