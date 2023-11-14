from time import sleep

import numpy as np


class WeightDiffusionMapGenerator:
    def __init__(self, width: int, height: int) -> None:
        """
        Initialises the map generator, represented by two numpy 2D arrays of the shape (width, height)

        One array holds boolean values corresponding to the state of each square in the grid (where empty spaces are
        flagged with True and walls are flagged with False) while the other array holds weights which are needed to
        determine the next square to be 'dug out' in the sequential process
        """

        self.__empty = np.full(shape=(width, height), fill_value=False, order='F', dtype=bool)
        self.__weights = np.full(shape=(width, height), fill_value=0, order='F', dtype=int)

        # The width and height are stored for a more convenient access
        self.__width, self.__height = width, height

        self.__init_middle_slice()

    def __init_middle_slice(self):
        """
        Set the middle 2x2 slice of is_empty_grid to True and set the weights of the square "ring" around the Trues to 1
        """
        start_x = self.__width // 2 - 1
        end_x = start_x + 2

        start_y = self.__height // 2 - 1
        end_y = start_y + 2

        self.__empty[start_x:end_x, start_y:end_y] = True

        for x in range(start_x - 1, end_x + 1):
            for y in range(start_y - 1, end_y + 1):
                if not (start_x <= x < end_x and start_y <= y < end_y):
                    self.__weights[x, y] = 1

    def __str__(self) -> str:
        """
        Returns a string representation of the GameMap, using '#' for False and '.' for True
        """
        return '\n'.join(''.join('#' if not value else '.' for value in col) for col in self.__empty.T)

    @property
    def weights(self) -> str:
        """
        Returns a string representation of the weight_grid
        """
        return '\n'.join(' '.join(str(value) for value in col) for col in self.__weights.T)

    def dig_next_square(self) -> None:
        """
        Randomly selects a square based on weights, sets it to True, and updates weights for adjacent False squares
        """
        flattened_weights = self.__weights.flatten(order='F').astype(float)
        flattened_weights /= flattened_weights.sum()  # Normalize weights to probabilities

        # Randomly select an index based on weights
        chosen_index = np.random.choice(np.arange(len(flattened_weights)), p=flattened_weights)

        # Convert the flattened index to 2D coordinates
        x, y = np.unravel_index(chosen_index, self.__weights.shape, order='F')

        # Set the selected square to True
        self.__empty[x, y] = True

        self.__update_weights(x, y)

    def __update_weights(self, x: int, y: int) -> None:
        """
        Update weights for all adjacent False squares to 1 and set the current square's weight to 0
        """
        self.__weights[x, y] = 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                offset_x, offset_y = x + dx, y + dy
                in_bounds_new_x = (0 <= offset_x < self.__weights.shape[0])
                in_bounds_new_y = (0 <= offset_y < self.__weights.shape[1])
                if in_bounds_new_x and in_bounds_new_y and not self.__empty[offset_x, offset_y]:
                    if self.__weights[offset_x, offset_y] == 0:
                        self.__weights[offset_x, offset_y] = 1



generator = WeightDiffusionMapGenerator(96, 54)

print(generator)

for _ in range(1500):
    generator.dig_next_square()

    print(generator)
    print('|'*96)
    sleep(0.001)
