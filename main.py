from time import sleep

import numpy as np


class GameMap:
    def __init__(self, width: int, height: int) -> None:
        self.is_empty_grid = np.full(shape=(width, height), fill_value=False, order='F', dtype=bool)
        self.weight_grid = np.full(shape=(width, height), fill_value=0, order='F', dtype=int)

        # Set the middle 2x2 slice of is_empty_grid to True
        middle_start_x = width // 2 - 1
        middle_end_x = middle_start_x + 2
        middle_start_y = height // 2 - 1
        middle_end_y = middle_start_y + 2
        self.is_empty_grid[middle_start_x:middle_end_x, middle_start_y:middle_end_y] = True

        # Set the weights of the square "ring" around the Trues to 1
        for x in range(middle_start_x - 1, middle_end_x + 1):
            for y in range(middle_start_y - 1, middle_end_y + 1):
                if not (middle_start_x <= x < middle_end_x and middle_start_y <= y < middle_end_y):
                    self.weight_grid[x, y] = 1

    def __str__(self) -> str:
        """
        Returns a string representation of the GameMap, using '#' for False and '.' for True
        """
        return '\n'.join(''.join('#' if not value else '.' for value in col) for col in self.is_empty_grid.T)

    @property
    def weights(self) -> str:
        """
        Returns a string representation of the weight_grid
        """
        return '\n'.join(''.join(str(value) for value in col) for col in self.weight_grid.T)

    def dig_next_square(self) -> None:
        """
        Randomly selects a square based on weights, sets it to True, and updates weights for adjacent False squares
        """
        flattened_weights = self.weight_grid.flatten(order='F').astype(float)
        flattened_weights /= flattened_weights.sum()  # Normalize weights to probabilities

        # Randomly select an index based on weights
        chosen_index = np.random.choice(np.arange(len(flattened_weights)), p=flattened_weights)

        # Convert the flattened index to 2D coordinates
        x, y = np.unravel_index(chosen_index, self.weight_grid.shape, order='F')

        # Set the selected square to True
        self.is_empty_grid[x, y] = True

        self.__update_weights(x, y)

    def __update_weights(self, x: int, y: int) -> None:
        """
        Update weights for all adjacent False squares to 1 and set the current square's weight to 0
        """
        self.weight_grid[x, y] = 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                in_bounds_new_x = 0 <= new_x < self.weight_grid.shape[0]
                in_bounds_new_y = 0 <= new_y < self.weight_grid.shape[1]
                if in_bounds_new_x and in_bounds_new_y and not self.is_empty_grid[new_x, new_y]:
                    self.weight_grid[new_x, new_y] = 1


gm = GameMap(96, 54)

print(gm.is_empty_grid.shape)
print(gm.weights)

for _ in range(1000):
    gm.dig_next_square()

    print(gm.weights)

    sleep(0.01)
