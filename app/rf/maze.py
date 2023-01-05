import numpy as np
import random


class Maze:
    """The maze should only be created via classmethods
    """
    def __init__(self, walls, rewards, goal_states = None):
        self.walls = walls
        self.rewards = rewards
        goals_x, goals_y = np.where(rewards > 0)
        self.goal_states = [(x, y) for x, y in zip(goals_x, goals_y)]

    @property
    def shape(self):
        return self.walls.shape

    @classmethod
    def load_from_numpy_array(cls, file):
        walls, rewards = np.load(file)
        return cls(walls, rewards)

    def from_json(self):
        pass

    def __getitem__(self, *args):
        return self.walls.__getitem__(*args)


class MazeGenerator:
    def __init__(self, seed = None):
        self.seed = seed
        if seed:
            random.seed(seed)

    def _generate_prims(self, shape):
        pass

    def generate(self, i, j):
        maze = np.full((i, j), True, dtype=bool)

        # Choose a random starting point
        start_row = random.randint(0, i-1)
        start_col = random.randint(0, j-1)
        maze[start_row][start_col] = False

        # Add starting point to the list of walls to consider
        walls = [(start_row, start_col)]

        # While there are still walls to consider
        while walls:
            # Choose a random wall from the list
            wall_index = random.randint(0, len(walls)-1)
            wall_row, wall_col = walls.pop(wall_index)

            # Find all adjacent cells to the wall
            adjacent_cells = []
            if wall_row > 0:
                adjacent_cells.append((wall_row-1, wall_col))
            if wall_row < i-1:
                adjacent_cells.append((wall_row+1, wall_col))
            if wall_col > 0:
                adjacent_cells.append((wall_row, wall_col-1))
            if wall_col < j-1:
                adjacent_cells.append((wall_row, wall_col+1))

            # Choose a random adjacent cell
            cell_index = random.randint(0, len(adjacent_cells)-1)
            cell_row, cell_col = adjacent_cells[cell_index]

            # If the adjacent cell has not been visited yet
            if maze[cell_row][cell_col]:
                # Remove the wall between the cell and the wall
                maze[wall_row][wall_col] = False
                # Mark the cell as visited
                maze[cell_row][cell_col] = False
                # Add the cell's walls to the list of walls to consider
                if cell_row > 0:
                    walls.append((cell_row-1, cell_col))
                if cell_row < i-1:
                    walls.append((cell_row+1, cell_col))
                if cell_col > 0:
                    walls.append((cell_row, cell_col-1))
                if cell_col < j-1:
                    walls.append((cell_row, cell_col+1))

        return maze
