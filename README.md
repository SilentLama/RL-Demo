# Reinforcement Learning demo

This is an attempt to visualize the DynaQ model in different environments (currently just mazes) using pygame.

## How to start

There are some basic scenes that are already implemented that can be imported. Every scene has a pause-slider that indicates the duration of a real world step the agent takes.

This loads a basic scene with a standard DynaQ agent in a 10x10 maze generated using the binary_tree algorithm:

```py
from app.scene import DynaQMazeScene
from app.rf.maze import MazeGenerator

DynaQMazeScene(1920, 1080, "DynaQ-Demo", MazeGenerator.generate(10, 15, algorithm="binary_tree", goal_reward=1)).run()
```

The following script loads a scene with three agents that can be individually parametrized in regards to their planning.

```py
from app.scene import DynaQMazeMultiAgentScene
from app.rf.maze import MazeGenerator

DynaQMazeMultiAgentScene(1920, 1080, "DynaQ-Demo", MazeGenerator.generate(10, 15, algorithm="prims", goal_reward=1)).run()
```