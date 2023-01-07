from app.scene import DynaQMazeScene, DynaQMazeMultiAgentScene
from app.rf.maze import Maze, MazeGenerator

if __name__ == "__main__":
    # DynaQMazeScene(1920, 1080, "DynaQ-Demo", Maze.load_from_numpy_array("./mazes/first_example_maze_positive_reward.npy", (2, 0))).run()
    # DynaQMazeScene(1920, 1080, "DynaQ-Demo", MazeGenerator.generate(10, 14)).run()
    DynaQMazeMultiAgentScene(1920, 1080, "DynaQ-Multi-Agent-Demo", MazeGenerator.generate(10, 14)).run()