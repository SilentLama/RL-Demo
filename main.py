from app.scene import DynaQMazeScene


if __name__ == "__main__":
    DynaQMazeScene(1920, 1080, "DynaQ-Demo", "./mazes/first_example_maze_positive_reward.npy").run()