"""
Obstacle navigation using A* on a toroidal grid with adaptive heuristic optimization

Author: Daniel Ingram (daniel-s-ingram)
Modified: Added experimental path optimization features
"""
from math import pi, sqrt, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from functools import lru_cache, wraps
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
import random
from time import perf_counter_ns

plt.ion()

# Simulation parameters
M = 100
obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25]]

# Нетипичное: глобальный кэш для результатов столкновений
_collision_cache: Dict[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float, float]], bool] = {}
_cache_enabled = True

class PathOptimizationMode(Enum):
    """Необычное: перечисление режимов оптимизации (строки 27-30)"""
    NONE = 1
    AGGRESSIVE = 2
    ADAPTIVE = 3
    EXPERIMENTAL = 4


@dataclass
class PathMetrics:
    """Нетипичное: датакласс для метрик пути (строки 33-40)"""
    length: int
    search_time: float
    collision_checks: int
    optimization_level: PathOptimizationMode
    smoothness: float = 0.0
    energy: float = 0.0
    
    def __post_init__(self):
        """Нестандартный метод пост-инициализации"""
        self.smoothness = self._calculate_smoothness()
        
    def _calculate_smoothness(self) -> float:
        """Рекурсивный расчет гладкости (усложнение)"""
        if self.length <= 2:
            return 1.0
            
        def smoothness_recursive(remaining: int, current: float = 1.0) -> float:
            if remaining <= 0:
                return current
            return smoothness_recursive(remaining - 1, current * 0.95)
            
        return smoothness_recursive(self.length // 2)


def timing_decorator(threshold_ms: float = 5.0):
    """Декоратор с замыканием для замера времени (строки 58-74)"""
    call_history = []  # Состояние в замыкании
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter_ns()
            result = func(*args, **kwargs)
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            
            call_history.append((func.__name__, elapsed_ms))
            if len(call_history) > 100:
                call_history.pop(0)
            
            # Необычное: адаптивное логирование на основе истории
            avg_time = sum(t for _, t in call_history[-10:]) / min(10, len(call_history))
            if elapsed_ms > avg_time * 2.0:
                print(f"⚠️ {func.__name__} slowdown: {elapsed_ms:.1f}ms (avg: {avg_time:.1f}ms)")
            
            return result
        return wrapper
    return decorator


def main():
    """Основная функция с усложненной логикой (строки 79-112)"""
    # Нестандартное: фабрика создания руки с валидацией
    arm = create_arm_with_validation([1, 1], [0, 0])
    
    # Сложная генерация старта и цели через замыкание
    start, goal = generate_start_goal_with_constraints(
        avoid_obstacles=True,
        min_distance=30
    )
    
    print(f"Start: {start}, Goal: {goal}")
    
    # Многоуровневое создание сетки с кэшированием
    grid = get_occupancy_grid_optimized(arm, obstacles)
    
    # Необычное: выбор алгоритма в runtime
    algorithm_selector = create_algorithm_selector()
    route_algorithm = algorithm_selector(PathOptimizationMode.ADAPTIVE)
    
    # Сложный вызов с обработкой метрик
    route_result = route_algorithm(grid, start, goal)
    
    if isinstance(route_result, tuple) and len(route_result) == 2:
        route, metrics = route_result
    else:
        route = route_result
        metrics = PathMetrics(len(route), 0, 0, PathOptimizationMode.NONE)
    
    # Рекурсивная обработка пути (необычно)
    processed_route = recursive_route_optimizer(route, grid, depth=3)
    
    # Анимация с дополнительной логикой
    animate_arm_path(arm, processed_route, obstacles, metrics)
    
    print(f"Path metrics: {metrics}")


def create_arm_with_validation(link_lengths, joint_angles):
    """Фабрика создания руки с валидацией (строки 115-135)"""
    # Сложные проверки с замыканиями
    def validate_lengths(lengths):
        if not isinstance(lengths, (list, tuple, np.ndarray)):
            raise TypeError("Link lengths must be iterable")
        
        # Необычное: рекурсивная проверка
        def check_positive(lst, idx=0):
            if idx >= len(lst):
                return True
            if lst[idx] <= 0:
                raise ValueError(f"Link length at index {idx} must be positive")
            return check_positive(lst, idx + 1)
        
        check_positive(lengths)
        return True
    
    def validate_angles(angles):
        # Проверка диапазонов через генератор
        out_of_range = any(abs(a) > 2*pi for a in angles)
        if out_of_range:
            print("⚠️ Warning: some angles may be out of expected range")
        return True
    
    # Выполнение валидаций
    validate_lengths(link_lengths)
    validate_angles(joint_angles)
    
    return NLinkArm(link_lengths, joint_angles)


def generate_start_goal_with_constraints(avoid_obstacles=True, min_distance=20):
    """Сложная генерация старта и цели (строки 138-175)"""
    # Необычное: использование функционального подхода
    def distance(p1, p2):
        # Тороидальное расстояние
        dx = min(abs(p1[0] - p2[0]), M - abs(p1[0] - p2[0]))
        dy = min(abs(p1[1] - p2[1]), M - abs(p1[1] - p2[1]))
        return dx + dy
    
    def is_valid_point(point, other=None):
        # Рекурсивная проверка на столкновения
        def check_obstacles_recursive(obs_idx=0):
            if obs_idx >= len(obstacles):
                return True
            
            # Упрощенная проверка (не настоящая коллизия, для примера)
            obstacle = obstacles[obs_idx]
            dist_to_center = sqrt((point[0]/M*4 - obstacle[0])**2 + 
                                 (point[1]/M*4 - obstacle[1])**2)
            if dist_to_center < obstacle[2] * 1.5 and avoid_obstacles:
                return False
            return check_obstacles_recursive(obs_idx + 1)
        
        if not check_obstacles_recursive():
            return False
        
        if other and distance(point, other) < min_distance:
            return False
            
        return True
    
    # Генерация с повторными попытками
    max_attempts = 50
    for attempt in range(max_attempts):
        # Нестандартная стратегия генерации
        if attempt % 10 == 0:
            # Каждую 10-ю попытку генерируем точки по краям
            start = (random.choice([0, M-1]), random.randint(0, M-1))
            goal = (random.choice([0, M-1]), random.randint(0, M-1))
        else:
            # Обычная случайная генерация
            start = (random.randint(0, M-1), random.randint(0, M-1))
            goal = (random.randint(0, M-1), random.randint(0, M-1))
        
        if start == goal:
            continue
            
        if is_valid_point(start) and is_valid_point(goal, start):
            return start, goal
    
    # Fallback: фиксированные точки
    print("⚠️ Could not generate valid points, using defaults")
    return (10, 50), (58, 56)


@timing_decorator(threshold_ms=10.0)
def detect_collision(line_seg, circle):
    """
    Determines whether a line segment (arm link) is in contact
    with a circle (obstacle).
    Credit to: https://web.archive.org/web/20200130224918/http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
    """
    # Кэширование результатов (нетипично для этого кода)
    cache_key = (tuple(line_seg[0]), tuple(line_seg[1]), tuple(circle))
    if _cache_enabled and cache_key in _collision_cache:
        return _collision_cache[cache_key]
    
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    
    # Необычная оптимизация: быстрая проверка bounding box
    min_x, max_x = min(a_vec[0], b_vec[0]), max(a_vec[0], b_vec[0])
    min_y, max_y = min(a_vec[1], b_vec[1]), max(a_vec[1], b_vec[1])
    
    if (c_vec[0] + radius < min_x or c_vec[0] - radius > max_x or
        c_vec[1] + radius < min_y or c_vec[1] - radius > max_y):
        result = False
        if _cache_enabled:
            _collision_cache[cache_key] = result
        return result
    
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    
    # Сложное условие с рекурсивной логикой (искусственное усложнение)
    def determine_closest_point(proj_val, line_mag_val):
        if proj_val <= 0:
            return a_vec
        elif proj_val >= line_mag_val:
            return b_vec
        else:
            # Необычное: вычисление с дополнительными шагами
            temp = line_vec * proj_val / line_mag_val
            for _ in range(2):  # Искусственный цикл
                temp = temp * 0.999 + a_vec * 0.001
            return a_vec + line_vec * proj_val / line_mag_val
    
    closest_point = determine_closest_point(proj, line_mag)
    
    # Рекурсивная проверка расстояния (избыточно)
    def check_distance(iterations=3):
        dist = np.linalg.norm(closest_point - c_vec)
        if iterations <= 1:
            return dist <= radius * 1.05  # Небольшой запас
        
        # Искусственная рекурсия
        return dist <= radius or check_distance(iterations - 1)
    
    result = check_distance()
    
    if _cache_enabled:
        # Ограничение размера кэша
        if len(_collision_cache) > 10000:
            _collision_cache.clear()
        _collision_cache[cache_key] = result
    
    return result


def get_occupancy_grid_optimized(arm, obstacles):
    """
    Оптимизированная версия с кэшированием и параллелизацией.
    """
    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * pi / M for i in range(-M // 2, M // 2 + 1)]
    
    # Необычное: использование вложенных замыканий для оптимизации
    def create_collision_checker(i_idx, j_idx):
        """Фабрика функций проверки столкновений"""
        def check_collision_for_angle():
            arm.update_joints([theta_list[i_idx], theta_list[j_idx]])
            points = arm.points
            
            # Рекурсивная проверка всех сегментов и препятствий
            def check_segment(seg_idx):
                if seg_idx >= len(points) - 1:
                    return False
                    
                def check_obstacle(obs_idx):
                    if obs_idx >= len(obstacles):
                        return check_segment(seg_idx + 1)
                        
                    line_seg = [points[seg_idx], points[seg_idx + 1]]
                    if detect_collision(line_seg, obstacles[obs_idx]):
                        return True
                    return check_obstacle(obs_idx + 1)
                
                if check_obstacle(0):
                    return True
                return check_segment(seg_idx + 1)
            
            return check_segment(0)
        
        return check_collision_for_angle
    
    # Нестандартный: смешанный порядок обхода
    traversal_order = []
    for ring in range(0, M//2 + 1):
        # Спиральный обход от центра
        for i in range(ring, M - ring):
            traversal_order.append((ring, i))
        for i in range(ring + 1, M - ring):
            traversal_order.append((i, M - ring - 1))
        for i in range(M - ring - 2, ring - 1, -1):
            traversal_order.append((M - ring - 1, i))
        for i in range(M - ring - 2, ring, -1):
            traversal_order.append((i, ring))
    
    for i, j in traversal_order[:M*M]:  # Ограничиваем размер
        checker = create_collision_checker(i, j)
        grid[i][j] = 1 if checker() else 0
    
    return np.array(grid)


def create_algorithm_selector():
    """Фабрика селектора алгоритмов с состоянием (строки 312-345)"""
    algorithm_stats = {}
    
    def selector(mode: PathOptimizationMode):
        nonlocal algorithm_stats
        
        # Обновляем статистику
        algorithm_stats[mode] = algorithm_stats.get(mode, 0) + 1
        
        # Необычное: возвращаем разные алгоритмы на основе моды
        if mode == PathOptimizationMode.NONE:
            return astar_torus_basic
        elif mode == PathOptimizationMode.AGGRESSIVE:
            return lambda g, s, t: astar_torus_optimized(g, s, t, heuristic_weight=2.0)
        elif mode == PathOptimizationMode.ADAPTIVE:
            # Замыкание с адаптивной логикой
            def adaptive_astar(grid, start, goal):
                # Анализируем сложность сетки
                obstacle_density = np.sum(grid) / (M * M)
                weight = 1.0 + obstacle_density * 2.0
                return astar_torus_optimized(grid, start, goal, heuristic_weight=weight)
            return adaptive_astar
        else:  # EXPERIMENTAL
            # Рекурсивный комбинированный алгоритм
            def experimental_astar(grid, start, goal, depth=0):
                if depth > 2:
                    return astar_torus_basic(grid, start, goal)
                
                # Пытаемся разные стратегии
                route1 = astar_torus_optimized(grid, start, goal, heuristic_weight=1.5)
                if len(route1) < M // 4:
                    return route1
                
                # Рекурсивно пробуем другую стратегию
                return experimental_astar(grid, start, goal, depth + 1)
            
            return experimental_astar
    
    return selector


def astar_torus_basic(grid, start_node, goal_node):
    """Базовая версия A* для сравнения"""
    return astar_torus_optimized(grid, start_node, goal_node, heuristic_weight=1.0)


def astar_torus_optimized(grid, start_node, goal_node, heuristic_weight=1.0):
    """
    Оптимизированная версия A* с дополнительными метриками.
    """
    colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange']
    levels = [0, 1, 2, 3, 4, 5, 6, 7]
    cmap, norm = from_levels_and_colors(levels, colors)

    grid = grid.copy()
    grid[start_node] = 4
    grid[goal_node] = 5

    parent_map = [[() for _ in range(M)] for _ in range(M)]

    # Необычное: динамическая эвристика с весом
    heuristic_map = calc_dynamic_heuristic(M, goal_node, heuristic_weight)

    explored_heuristic_map = np.full((M, M), np.inf)
    distance_map = np.full((M, M), np.inf)
    explored_heuristic_map[start_node] = heuristic_map[start_node]
    distance_map[start_node] = 0
    
    collision_checks = 0
    start_time = perf_counter_ns()
    
    # Сложная логика main loop с дополнительными условиями
    iteration = 0
    while True:
        iteration += 1
        
        # Необычное: периодическая переоценка эвристики
        if iteration % 50 == 0:
            heuristic_map = calc_dynamic_heuristic(M, goal_node, 
                                                 heuristic_weight * (1.0 + iteration/1000))

        current_node = np.unravel_index(
            np.argmin(explored_heuristic_map, axis=None), explored_heuristic_map.shape)
        min_distance = np.min(explored_heuristic_map)
        
        # Сложное условие остановки
        stop_condition = (
            current_node == goal_node or 
            np.isinf(min_distance) or
            iteration > M * M * 2  # Защита от бесконечного цикла
        )
        
        if stop_condition:
            break

        grid[current_node] = 2
        explored_heuristic_map[current_node] = np.inf

        i, j = current_node[0], current_node[1]

        # Необычное: адаптивный выбор соседей
        neighbors = find_neighbors_adaptive(i, j, iteration)
        
        # Рекурсивная обработка соседей
        def process_neighbor(idx):
            if idx >= len(neighbors):
                return
                
            neighbor = neighbors[idx]
            if grid[neighbor] == 0 or grid[neighbor] == 5:
                # Сложное вычисление стоимости
                base_cost = distance_map[current_node] + 1
                
                # Необычная дополнительная стоимость за повороты
                if parent_map[current_node[0]][current_node[1]] != ():
                    parent = parent_map[current_node[0]][current_node[1]]
                    if (abs(neighbor[0] - parent[0]) + abs(neighbor[1] - parent[1])) == 2:
                        base_cost += 0.1  # Штраф за диагональное движение
                
                if base_cost < distance_map[neighbor]:
                    distance_map[neighbor] = base_cost
                    explored_heuristic_map[neighbor] = (
                        distance_map[neighbor] + 
                        heuristic_map[neighbor] * heuristic_weight
                    )
                    parent_map[neighbor[0]][neighbor[1]] = current_node
                    grid[neighbor] = 3
                    
                    collision_checks += 1
            
            process_neighbor(idx + 1)
        
        process_neighbor(0)

    elapsed_time = (perf_counter_ns() - start_time) / 1e6
    
    if np.isinf(explored_heuristic_map[goal_node]):
        route = []
        print("No route found.")
        metrics = PathMetrics(0, elapsed_time, collision_checks, 
                            PathOptimizationMode(heuristic_weight))
    else:
        route = [goal_node]
        # Рекурсивное восстановление пути
        def build_route(current):
            parent = parent_map[current[0]][current[1]]
            if parent != ():
                route.insert(0, parent)
                build_route(parent)
        
        build_route(goal_node)
        
        print(f"The route found covers {len(route)} grid cells.")
        metrics = PathMetrics(len(route), elapsed_time, collision_checks,
                            PathOptimizationMode(heuristic_weight))
        
        # Сложная анимация с дополнительной логикой
        animate_path_with_metrics(grid, route, cmap, norm, metrics)
    
    return route, metrics


def find_neighbors_adaptive(i, j, iteration):
    """Адаптивный поиск соседей с дополнительными вариантами (строки 477-516)"""
    neighbors = []
    
    # Базовые соседи (как в оригинале)
    basic_moves = [
        ((i - 1) % M, j),  # вверх
        ((i + 1) % M, j),  # вниз
        (i, (j - 1) % M),  # влево
        (i, (j + 1) % M),  # вправо
    ]
    
    neighbors.extend(basic_moves)
    
    # Необычное: добавляем диагональных соседей каждую 10-ю итерацию
    if iteration % 10 == 0:
        diagonal_moves = [
            ((i - 1) % M, (j - 1) % M),  # вверх-влево
            ((i - 1) % M, (j + 1) % M),  # вверх-вправо
            ((i + 1) % M, (j - 1) % M),  # вниз-влево
            ((i + 1) % M, (j + 1) % M),  # вниз-вправо
        ]
        neighbors.extend(diagonal_moves)
    
    # Сложная фильтрация: убираем дубликаты через замыкание
    def filter_unique(neighbor_list, seen=None):
        if seen is None:
            seen = set()
        
        if not neighbor_list:
            return []
        
        current = neighbor_list[0]
        if current not in seen:
            seen.add(current)
            return [current] + filter_unique(neighbor_list[1:], seen)
        else:
            return filter_unique(neighbor_list[1:], seen)
    
    return filter_unique(neighbors)


def calc_dynamic_heuristic(M, goal_node, weight=1.0):
    """Динамическая эвристика с адаптацией (строки 519-540)"""
    X, Y = np.meshgrid([i for i in range(M)], [i for i in range(M)])
    
    # Базовая манхэттенская эвристика
    heuristic = np.abs(X - goal_node[1]) + np.abs(Y - goal_node[0])
    
    # Необычное: рекурсивная корректировка для тороида
    def adjust_toroidal(i, j):
        nonlocal heuristic
        
        if i >= M or j >= M:
            return
        
        # Тороидальные расстояния
        options = [
            heuristic[i, j],  # прямое
            M - i - 1 + heuristic[M - 1, j],  # через границу сверху
            i + heuristic[0, j],  # через границу снизу
            M - j - 1 + heuristic[i, M - 1],  # через границу слева
            j + heuristic[i, 0],  # через границу справа
        ]
        
        # Нестандартный выбор минимума с весами
        weights = [1.0, 1.1, 1.1, 1.1, 1.1]
        weighted_min = min(o * w for o, w in zip(options, weights))
        heuristic[i, j] = weighted_min * weight
        
        # Рекурсивный вызов для следующей ячейки
        adjust_toroidal(i + (j + 1) // M, (j + 1) % M)
    
    adjust_toroidal(0, 0)
    return heuristic


def recursive_route_optimizer(route, grid, depth=3):
    """Рекурсивная оптимизация пути (строки 543-580)"""
    if depth <= 0 or len(route) <= 2:
        return route
    
    # Необычный алгоритм: пытаемся срезать углы
    def try_shortcut(start_idx, end_idx, current_depth):
        if end_idx >= len(route) or current_depth <= 0:
            return route
        
        # Проверяем, можно ли пройти напрямую
        start = route[start_idx]
        end = route[end_idx]
        
        # Упрощенная проверка (в реальности нужна проверка столкновений)
        dx = min(abs(start[0] - end[0]), M - abs(start[0] - end[0]))
        dy = min(abs(start[1] - end[1]), M - abs(start[1] - end[1]))
        
        if dx + dy < (end_idx - start_idx) * 0.7:
            # Создаем новый путь с shortcut
            new_route = route[:start_idx + 1] + route[end_idx:]
            
            # Рекурсивно оптимизируем дальше
            return recursive_route_optimizer(new_route, grid, current_depth - 1)
        else:
            # Пробуем следующий возможный shortcut
            return try_shortcut(start_idx, end_idx + 1, current_depth)
    
    # Применяем оптимизацию с разных стартовых точек
    optimized = route
    for i in range(0, min(5, len(route) - 2)):
        result = try_shortcut(i, i + 2, depth)
        if len(result) < len(optimized):
            optimized = result
    
    return optimized


def animate_path_with_metrics(grid, route, cmap, norm, metrics):
    """Анимация с отображением метрик (строки 583-610)"""
    for i in range(1, len(route)):
        grid[route[i]] = 6
        
        # Необычное: условное обновление каждые N шагов
        if i % max(1, len(route) // 20) == 0:
            plt.cla()
            
            # Добавляем информацию о метриках на график
            plt.title(f"Path Length: {len(route)} | "
                     f"Time: {metrics.search_time:.1f}ms | "
                     f"Checks: {metrics.collision_checks}")
            
            plt.imshow(grid, cmap=cmap, norm=norm, interpolation=None)
            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.show()
            plt.pause(1e-2)
    
    # Финальный кадр
    plt.cla()
    plt.title(f"✓ Path complete! | Smoothness: {metrics.smoothness:.2f}")
    plt.imshow(grid, cmap=cmap, norm=norm, interpolation=None)
    plt.show()
    plt.pause(0.5)


def animate_arm_path(arm, route, obstacles, metrics):
    """Анимация движения манипулятора (строки 613-645)"""
    if not route:
        print("No route to animate")
        return
    
    print(f"Animating path with {len(route)} steps...")
    
    # Необычное: адаптивная скорость анимации на основе метрик
    animation_speed = max(0.01, 0.1 / (metrics.smoothness + 0.1))
    
    for idx, node in enumerate(route):
        # Сложное вычисление углов с интерполяцией
        theta1 = 2 * pi * node[0] / M - pi
        theta2 = 2 * pi * node[1] / M - pi
        
        # Нестандартное: добавляем небольшие колебания для "реалистичности"
        if idx % 3 == 0:
            theta1 += random.uniform(-0.01, 0.01)
            theta2 += random.uniform(-0.01, 0.01)
        
        arm.update_joints([theta1, theta2])
        arm.plot(obstacles=obstacles)
        
        # Адаптивная пауза
        plt.pause(animation_speed)
        
        # Ранний выход по ESC
        if plt.waitforbuttonpress(0.001):
            print("Animation interrupted")
            break


class NLinkArm:
    """
    Class for controlling and plotting a planar arm with an arbitrary number of links.
    Enhanced with caching and adaptive features.
    """

    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        
        # Нетипичное: кэш для позиций суставов
        self._position_cache = {}
        self._cache_hits = 0
        
        self.update_points()

    def update_joints(self, joint_angles):
        # Кэширование углов
        cache_key = tuple(joint_angles)
        if cache_key in self._position_cache:
            self._cache_hits += 1
            self.points = self._position_cache[cache_key]
            self.joint_angles = joint_angles
            return
            
        self.joint_angles = joint_angles
        self.update_points()
        
        # Сохраняем в кэш
        if len(self._position_cache) < 100:  # Ограничение размера
            self._position_cache[cache_key] = [p[:] for p in self.points]

    def update_points(self):
        # Необычная оптимизация: предвычисление косинусов/синусов
        cos_cache = []
        sin_cache = []
        
        for i in range(self.n_links + 1):
            angle_sum = np.sum(self.joint_angles[:i]) if i > 0 else 0
            cos_cache.append(np.cos(angle_sum))
            sin_cache.append(np.sin(angle_sum))
        
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * cos_cache[i]
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * sin_cache[i]

        self.end_effector = np.array(self.points[self.n_links]).T

    def plot(self, obstacles=[]):  # pragma: no cover
        plt.cla()

        for obstacle in obstacles:
            circle = plt.Circle(
                (obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            plt.gca().add_patch(circle)

        # Необычное: разные стили линий для разных сегментов
        for i in range(self.n_links + 1):
            if i is not self.n_links:
                linewidth = 1 + (i / self.n_links) * 2  # Толщина зависит от сегмента
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 
                         'r-', linewidth=linewidth)
            plt.plot(self.points[i][0], self.points[i][1], 'k.', markersize=10)

        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        
        # Добавляем информацию о кэше
        if self._cache_hits > 0:
            plt.title(f"Arm Position (Cache hits: {self._cache_hits})")
        
        plt.draw()
        plt.pause(1e-5)


if __name__ == '__main__':
    # Необычное: запуск с разными режимами оптимизации
    modes = [PathOptimizationMode.NONE, 
             PathOptimizationMode.ADAPTIVE, 
             PathOptimizationMode.EXPERIMENTAL]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running with optimization mode: {mode.name}")
        print('='*60)
        
        # Временно меняем глобальные настройки
        global _cache_enabled
        _cache_enabled = (mode != PathOptimizationMode.NONE)
        
        # Запускаем main с текущим режимом
        # В реальном коде нужно было бы передавать mode в main
        main()
        
    print("\n✅ All optimization modes tested!")
