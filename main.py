import numpy as np
from collections import namedtuple
from time import sleep

Position = namedtuple("Position", "x y")


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

        # Initialise the map with one empty space in the middle
        self.__square_position = Position(self.__width // 2, self.__height // 2)
        self.__dig()

    def __str__(self) -> str:
        """
        Returns a string representation of the map, using '#' for walls and '.' for empty space
        """

        return '\n'.join(''.join('#' if not value else '.' for value in col) for col in self.__empty.T)

    @property
    def weights(self) -> str:
        """
        Returns a string representation of the weights
        """

        return '\n'.join(' '.join(str(value) for value in col) for col in self.__weights.T)

    def __update_square_neighbourhood_weights(self):
        """
        Sets current square's weight to 0, the square has been dug out and is no longer considered when choosing next
        square

        Initialises weights for adjacent wall squares which could be 'dug out' next (previously 0)
        """

        x, y = self.__square_position.x, self.__square_position.y

        self.__weights[x, y] = 0  #

        # For each neighbouring wall which is now 'touching' an empty square, set that wall's weight to 1 from 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                offset_x, offset_y = x + dx, y + dy
                in_bounds_offset_x = (0 <= offset_x < self.__weights.shape[0])
                in_bounds_offset_y = (0 <= offset_y < self.__weights.shape[1])
                in_bounds = in_bounds_offset_x and in_bounds_offset_y
                if in_bounds and not self.__empty[offset_x, offset_y]:
                    if self.__weights[offset_x, offset_y] == 0:
                        self.__weights[offset_x, offset_y] = 1

    def __update_weights(self) -> None:
        """
        Call the initial method to update neighbouring weights and update the remaining weights according to some
        algorithm
        """

        self.__update_square_neighbourhood_weights()

        # TODO: Update the remaining weights according to some algorithm

    def __dig(self):
        """
        Set the boolean array for the 'dug out' square and call the method to update weights
        """

        pos_x, pos_y = self.__square_position.x, self.__square_position.y

        self.__empty[pos_x, pos_y] = True

        self.__update_weights()

    def dig_next_square(self) -> None:
        """
        Randomly selects a wall square based on weights and 'digs it out'
        """
        
        flattened_weights = self.__weights.flatten(order='F').astype(float)
        flattened_weights /= flattened_weights.sum()  # Normalize weights to probabilities

        # Randomly select an index based on weights
        chosen_index = np.random.choice(np.arange(len(flattened_weights)), p=flattened_weights)

        # Convert the flattened index to 2D coordinates
        x, y = np.unravel_index(indices=chosen_index, shape=self.__weights.shape, order='F')

        self.__square_position = Position(x, y)

        self.__dig()


generator = WeightDiffusionMapGenerator(96, 54)

print(generator)

for _ in range(2500):
    generator.dig_next_square()

    print(generator)
    print('|'*96)
    sleep(0.001)
