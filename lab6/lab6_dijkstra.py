import numpy as np
import matplotlib.pyplot as plt
import heapq
import timeit


def dijkstra_grid(input_map, start_coords, dest_coords, draw_map=False):
    nrows, ncols = input_map.shape
    map = np.copy(input_map)
    map[start_coords] = 0
    map[dest_coords] = 0

    distance_from_start = np.full((nrows, ncols), np.inf)
    distance_from_start[start_coords] = 0
    parent = np.zeros((nrows, ncols), dtype=object)

    num_expanded = 0
    priority_queue = [(0, start_coords)]
    visited = set() #множество посещенных узлов

    while priority_queue:
        dist, current = heapq.heappop(priority_queue)
        if current == dest_coords:
            break
        if current in visited: #проверка на повторное посещение
            continue
        visited.add(current) #помечаем как посещенный
        map[current] = 2 #пометить как посещенную

        num_expanded += 1
        neighbors = get_neighbors(current, nrows, ncols, input_map)

        for neighbor in neighbors:
            new_dist = dist + 1
            if new_dist < distance_from_start[neighbor]:
                distance_from_start[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(priority_queue, (new_dist, neighbor))

        if draw_map:
            plt.imshow(map, cmap=plt.cm.gray)
            plt.title('Алгоритм Дейкстры')
            plt.pause(0.1)

    route = reconstruct_path(parent, start_coords, dest_coords)
    return route, num_expanded

def get_neighbors(node, nrows, ncols, input_map):
    row, col = node
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < nrows and 0 <= new_col < ncols and not input_map[new_row, new_col]:
            neighbors.append((new_row, new_col))
    return neighbors


def reconstruct_path(parent, start, end):
    path = []
    current = end
    while current is not None and not np.array_equal(current, start):
        path.append(current)
        try:
            current = parent[tuple(current)] # Преобразуем в кортеж перед использованием в качестве индекса
        except (IndexError, KeyError, TypeError): # Обрабатываем все возможные ошибки индексации
            return None # Путь не найден

    if current is not None:
        path.append(start)
    return path[::-1] if path else None

def generate_random_obstacles(rows, cols, obstacle_probability):
    return np.random.choice([0, 1], size=(rows, cols), p=[1 - obstacle_probability, obstacle_probability])


def visualize_path(input_map, route):
    if route is None:
        print("Путь не найден!")
        return 
    nrows, ncols = input_map.shape
    map = np.copy(input_map) # Создаем копию, чтобы не менять исходный масси

    for cell in route:
        map[cell] = 2  # Путь (красный)

    cmap = plt.cm.colors.ListedColormap(['gray', 'black', 'red']) # чёрный для препятствий
    plt.imshow(map, cmap=cmap, interpolation='nearest') # interpolation='nearest' для чёткого отображения
    plt.title('Алгоритм Дейкстры: Кратчайший путь')
    plt.show()

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
        route, num_expanded = dijkstra_grid(stable_micro_input_map, start_cell, micro_end_cell, draw_map=True)
        visualize_path(stable_micro_input_map, route)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    def micro_input_map_task():
        micro_end_cell = (4, 4)
        micro_input_map = generate_random_obstacles(5, 5, obstacle_probability)
        print(micro_input_map)
        route, num_expanded = dijkstra_grid(micro_input_map, start_cell, micro_end_cell, draw_map=True)
        visualize_path(micro_input_map, route)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()


    def normal_end_cell_task():
        normal_end_cell = (8, 9)
        normal_input_map = generate_random_obstacles(10, 10, obstacle_probability)
        print(normal_input_map)
        route, num_expanded = dijkstra_grid(normal_input_map, start_cell, normal_end_cell, draw_map=True)
        visualize_path(normal_input_map, route)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    def big_end_cell_task():
        big_end_cell=(13, 14)
        big_input_map = generate_random_obstacles(15, 15, obstacle_probability)
        print(big_input_map)
        route, num_expanded = dijkstra_grid(big_input_map, start_cell, big_end_cell, draw_map=True)
        visualize_path(big_input_map, route)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()

    
    def stable_normal_map_task():
        '''
        Test params
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
        route, num_expanded = dijkstra_grid(stable_test_input_map, test_start_cell, test_dest_cell, draw_map=True)
        print(route)
        visualize_path(stable_test_input_map, route)
        print("Кратчайший путь:", route)
        print("Количество обработанных узлов:", num_expanded)
        plt.show()


    execution_time = timeit.timeit(stable_normal_map_task, number=1)
    print(f"Время выполнения stable_normal_map_task: {execution_time:.4f} секунд")

    

    
    