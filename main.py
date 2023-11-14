import numpy as np
from collections import namedtuple
from typing import Set
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
        self.__current_position = Position(self.__width // 2, self.__height // 2)
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
        def symbol(value: int) -> str:
            if value == 0:
                return ' '
            elif value == 1:
                return 'I'
            elif value == 10:
                return 'X'
            elif value == 100:
                return 'D'
            elif value == 1000:
                return 'M'
            else:
                return '?'
        return '\n'.join(''.join(symbol(value) for value in col) for col in self.__weights.T)

    def __neighbouring_wall_positions(self, x: int, y: int) -> Set[Position]:
        wall_positions = set()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                offset_x, offset_y = x + dx, y + dy
                in_bounds_offset_x = (0 <= offset_x < self.__width)
                in_bounds_offset_y = (0 <= offset_y < self.__height)
                in_bounds = in_bounds_offset_x and in_bounds_offset_y
                if in_bounds and not self.__empty[offset_x, offset_y]:
                    wall_positions.add(Position(offset_x, offset_y))

        return wall_positions

    def __init_neighbouring_weights(self) -> Set[Position]:
        """
        Sets current square's weight to 0, the square has been dug out and is no longer considered when choosing next
        square

        Initialises weights for adjacent wall squares which could be 'dug out' next (previously 0)
        """

        x, y = self.__current_position.x, self.__current_position.y

        self.__weights[x, y] = 0

        wall_positions = self.__neighbouring_wall_positions(x, y)

        for wall_position in wall_positions:
            if self.__weights[wall_position.x, wall_position.y] == 0:
                self.__weights[wall_position.x, wall_position.y] = 1

        return wall_positions

    def __set_neighbouring_weights(self, adjacent_walls: Set[Position]) -> None:
        for adjacent_wall in adjacent_walls:
            x, y = adjacent_wall.x, adjacent_wall.y
            walls = self.__neighbouring_wall_positions(x, y)
            num_walls = len(walls) - 1  # We don't count the current wall

            if num_walls == 2 or num_walls == 7:  # 2,7
                self.__weights[x, y] = 10
            elif num_walls == 3 or num_walls == 6:  # 3,6
                self.__weights[x, y] = 100
            elif num_walls == 4 or num_walls == 5:  # 4,5
                self.__weights[x, y] = 1000

    def __update_weights(self) -> None:
        """
        Call the initial method to update neighbouring weights and update the remaining weights according to some
        algorithm
        """

        adjacent_walls = self.__init_neighbouring_weights()

        self.__set_neighbouring_weights(adjacent_walls)

    def __dig(self) -> None:
        """
        Set the boolean array for the 'dug out' square and call the method to update weights
        """

        x, y = self.__current_position.x, self.__current_position.y

        self.__empty[x, y] = True

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

        self.__current_position = Position(x, y)

        self.__dig()


w = 272
h = 71

generator = WeightDiffusionMapGenerator(w, h)

print(generator)

for _ in range(2000):
    generator.dig_next_square()

    print(generator)
    # print(generator.weights)
    print('|'*w)
    sleep(0)
