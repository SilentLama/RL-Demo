from app.scene import DynaQMazeScene


if __name__ == "__main__":
    DynaQMazeScene(1920, 1080, "DynaQ-Demo", "./mazes/20x20_maze.npy").run()