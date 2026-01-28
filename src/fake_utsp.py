import random
import math
from tsp_core import(
    generate_cities,
    compute_distance_matrix,
    tour_length,
    two_opt
)

def generate_random_heatmap(n, seed=None):
    if seed is not None:
        random.seed(seed)

    heatmap = [[0.0]*n for _ in range(n)]
    for i in range (n):
        for j in range(n):
            if i!= j:
                heatmap[i][j]=random.random()
    return heatmap

def decode_greedy(heatmap, start = 0):
    n = len(heatmap)
    visited = [False]*n
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n-1):
        next_city = None
        best_score = -float("inf")

        for j in range(n):
            if not visited[j] and heatmap[current][j]>best_score:
                best_score = heatmap[current
                                     ][j]
                next_city= j
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    return tour

if __name__ == "__main__":
    n = 20
    cities = generate_cities(n, seed=42)
    dist = compute_distance_matrix(cities)

    # Fake "GNN output"
    heatmap = generate_random_heatmap(n, seed=123)

    # Decode into a tour
    tour = decode_greedy(heatmap, start=0)
    length_before = tour_length(tour, dist)

    # Improve with 2-opt
    improved_tour = two_opt(tour, dist)
    length_after = tour_length(improved_tour, dist)

    print("Decoded tour length:", length_before)
    print("After 2-opt length: ", length_after)
