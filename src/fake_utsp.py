import random
import math

def generate_random_heatmap(n, seed=None):
    if seed is not None:
        random.seed(seed)
    heatmap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                heatmap[i][j] = random.random()
    return heatmap

def decode_greedy(heatmap, start=0):
    n = len(heatmap)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        next_city = None
        best_score = -float("inf")
        for j in range(n):
            if not visited[j] and heatmap[current][j] > best_score:
                best_score = heatmap[current][j]
                next_city = j
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    return tour