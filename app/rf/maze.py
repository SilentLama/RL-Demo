import numpy as np


class Maze:
    """The maze should only be created via classmethods
    """
    def __init__(self, walls, rewards, start):
        self.walls = walls
        self.rewards = rewards
        self.start = start
        goals_x, goals_y = np.where(rewards > 0)
        self.goal_states = [(x, y) for x, y in zip(goals_x, goals_y)]

    @property
    def shape(self):
        return self.walls.shape

    @classmethod
    def load_from_numpy_array(cls, file, start):
        walls, rewards = np.load(file)
        return cls(walls, rewards, start)

    def from_json(self):
        pass

    def __getitem__(self, *args):
        return self.walls.__getitem__(*args)


class MazeGenerator:

    @staticmethod
    def get_neighbour_cells(cell, x_max, y_max):
        neighbours = [(max(0, min(y_max, y)), max(0, min(x_max, x))) for (y, x) in MazeGenerator.get_surrounding_cells(*cell)]
        return [neighbour for neighbour in neighbours if neighbour != cell]

    @staticmethod
    def get_surrounding_cells(x, y):
        return (y, x + 1), (y + 1, x), (y, x - 1), (y - 1, x)

    @staticmethod
    def _generate_prims(rows, cols):
        maze = np.full((rows, cols), False)

        def get_surrounding_walls(x, y):
            return [cell for cell in MazeGenerator.get_neighbour_cells((x, y), cols - 1, rows - 1) if not maze[cell]]

        def get_number_of_path_neighbours(x, y):
            return sum(maze[wall] for wall in MazeGenerator.get_neighbour_cells((x, y), cols - 1, rows - 1))


        start_y, start_x = np.random.randint(1, rows - 1), np.random.randint(1, cols - 1)

        walls = []
        maze[start_y, start_x] = True
        for wall in ((start_y - 1, start_x), (start_y, start_x - 1), 
                    (start_y, start_x + 1), (start_y + 1, start_x)):
            walls.append(wall)

        for wall in walls:
            maze[wall] = True

        while walls:    
            wall_y, wall_x = walls.pop(np.random.randint(0, len(walls)))
            if ((wall_x not in (0, cols - 1) and not maze[wall_y, wall_x - 1] and maze[wall_y, wall_x + 1])
            or (wall_x not in (0, cols - 1) and not maze[wall_y, wall_x + 1] and maze[wall_y, wall_x - 1])
            or (wall_y not in (0, rows - 1) and not maze[wall_y - 1, wall_x] and maze[wall_y + 1, wall_x])
            or (wall_y not in (0, rows - 1) and not maze[wall_y + 1, wall_x] and maze[wall_y - 1, wall_x])):
                if get_number_of_path_neighbours(wall_x, wall_y) <= 1:
                    maze[wall_y, wall_x] = True
                    surrounding_walls = get_surrounding_walls(wall_x, wall_y)
                    for wall in surrounding_walls:
                        if wall not in walls: walls.append(wall)
                
            # if wall_x not in (0, cols - 1) and not maze[wall_y, wall_x + 1] and maze[wall_y, wall_x - 1]:
            #     if get_number_of_path_neighbours(wall_x, wall_y) <= 1:
            #         maze[wall_y, wall_x] = True
            #         surrounding_walls = get_surrounding_walls(wall_x, wall_y)
            #         for wall in surrounding_walls:
            #             if wall not in walls: walls.append(wall)
            
            # if wall_y not in (0, rows - 1) and not maze[wall_y - 1, wall_x] and maze[wall_y + 1, wall_x]:
            #     if get_number_of_path_neighbours(wall_x, wall_y) <= 1:
            #         maze[wall_y, wall_x] = True
            #         surrounding_walls = get_surrounding_walls(wall_x, wall_y)
            #         for wall in surrounding_walls:
            #             if wall not in walls: walls.append(wall)
            
            # if wall_y not in (0, rows - 1) and not maze[wall_y + 1, wall_x] and maze[wall_y - 1, wall_x]:
            #     if get_number_of_path_neighbours(wall_x, wall_y) <= 1:
            #         maze[wall_y, wall_x] = True
            #         surrounding_walls = get_surrounding_walls(wall_x, wall_y)
            #         for wall in surrounding_walls:
            #             if wall not in walls: walls.append(wall)
        return maze

    @staticmethod
    def generate_rewards(maze_walls, goal_state, base_reward = 0, goal_reward = 1):
        rewards = np.full(maze_walls.shape, base_reward)
        rewards[goal_state] = goal_reward
        return rewards
        
    
    @staticmethod
    def generate_start_and_goal(maze_walls, selection_chance = 1):
        y_max, x_max = maze_walls.shape[0] - 1, maze_walls.shape[1] - 1

        def search_path(*cell_stack):
            stack = []
            for cell in cell_stack:
                if maze_walls[cell] and np.random.rand() < selection_chance:
                    return cell
                for neighbour in MazeGenerator.get_neighbour_cells(cell, x_max, y_max):
                    stack.append(neighbour)
            return search_path(*cell_stack, *stack)

        corners = ((0, 0), (y_max, 0), (0, x_max), (y_max, x_max))
        start_corner = corners[np.random.randint(0, len(corners))]
        end_corner = corners[np.random.randint(0, len(corners))]
        while end_corner == start_corner:
            end_corner = corners[np.random.randint(0, len(corners))]
        return search_path(start_corner), search_path(end_corner)

    @staticmethod
    def generate(rows, cols, algorithm = "prims"):
        if algorithm.lower() == "prims":
            maze_walls = MazeGenerator._generate_prims(rows, cols)
            start, goal = MazeGenerator.generate_start_and_goal(maze_walls)
            rewards = MazeGenerator.generate_rewards(maze_walls, goal)
            return Maze(maze_walls, rewards, start) 
