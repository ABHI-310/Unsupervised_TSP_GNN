import random
import math

def generate_cities(n, seed=None):
    if seed is not None:
        random.seed(seed)
    return [(random.random(), random.random()) for _ in range(n)]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_distance_matrix(cities):
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = distance(cities[i], cities[j])
    return dist

def tour_length(tour, dist):
    length = 0.0
    n = len(tour)
    for i in range(n):
        length += dist[tour[i]][tour[(i + 1) % n]]
    return length

def generate_random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def two_opt(tour, dist):
    n = len(tour)
    improved = True

    while improved:
        improved = False
        best_length = tour_length(tour, dist)

        for i in range(n - 1):
            for k in range(i + 2, n):
                if k == n - 1 and i == 0:
                    continue
                new_tour = tour[:i + 1] + tour[i + 1: k + 1][::-1] + tour[k + 1:]
                new_length = tour_length(new_tour, dist)

                if new_length < best_length:
                    tour = new_tour
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break
    return tour