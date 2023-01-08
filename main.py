from app.scene import DynaQMazeScene, DynaQMazeMultiAgentScene, DynaQPlusMazeMultiAgentScene
from app.rf.maze import Maze, MazeGenerator

if __name__ == "__main__":
    # DynaQMazeScene(1920, 1080, "DynaQ-Demo", Maze.load_from_numpy_array("./mazes/first_example_maze_positive_reward.npy", (2, 0))).run()
    # DynaQMazeMultiAgentScene(1920, 1080, "DynaQ-Multi-Agent-Demo", MazeGenerator.generate(10, 18, goal_reward=1)).run()    
    DynaQPlusMazeMultiAgentScene(1920, 1080, "DynaQ+-Multi-Agent-Demo", Maze.load_from_numpy_array("./mazes/blocking_maze_example.npy", (5, 3))).run()