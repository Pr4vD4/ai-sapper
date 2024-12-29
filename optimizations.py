from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def calculate_state_vector(revealed_array, adjacent_mines_array, width, height):
    """Оптимизированная версия с параллельным выполнением"""
    state = np.zeros(width * height * 2, dtype=np.float32)
    
    for y in prange(height):  # Используем parallel range
        for x in range(width):
            idx = (y * width + x) * 2
            state[idx] = revealed_array[y, x]
            state[idx + 1] = adjacent_mines_array[y, x] / 8.0 if revealed_array[y, x] else 0.0
    return state

@jit(nopython=True, cache=True)
def calculate_adjacent_value(revealed_array, adjacent_mines_array, x, y, width, height):
    """Оптимизированная версия вычисления значения соседних клеток"""
    total_hints = 0
    revealed_neighbors = 0
    
    # Используем фиксированный список для оптимизации
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and revealed_array[ny, nx]:
            revealed_neighbors += 1
            total_hints += adjacent_mines_array[ny, nx]
    
    return 0.5 if revealed_neighbors == 0 else 1.0 - (total_hints / (revealed_neighbors * 8))

@jit(nopython=True, cache=True)
def find_safe_moves_fast(revealed_array, adjacent_mines_array, width, height):
    """Оптимизированная версия поиска безопасных ходов"""
    safe_moves = []
    for y in range(height):
        for x in range(width):
            if not revealed_array[y, x]:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < width and 0 <= ny < height and 
                            revealed_array[ny, nx] and adjacent_mines_array[ny, nx] == 0):
                            safe_moves.append((x, y))
                            break
                    if len(safe_moves) > 0 and safe_moves[-1] == (x, y):
                        break
    return safe_moves 