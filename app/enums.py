from enum import Enum

class RenderLayer(Enum):
    """An enumeration of the layers that game objects can be rendered on."""
    GAME = 1
    UI = 2