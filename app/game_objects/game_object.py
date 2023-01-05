from abc import ABC, abstractmethod

from ..enums import RenderLayer

class GameObject(ABC):
    def __init__(self, x, y, layer=RenderLayer.GAME):
        """An abstract base class for game objects.

        The game object is rendered using the `x` and `y` coordinates, and can be assigned to a layer using the `layer` parameter. The `update` and `draw` methods must be implemented by subclasses.

        :param int x: The x coordinate of the top left corner of the game object.
        :param int y: The y coordinate of the top left corner of the game object.
        :param Layer layer: The layer that the game object should be rendered on.
        """
        self.x = x
        self.y = y
        self.layer = layer

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
    
    @abstractmethod
    def update(self):
        """Update the game object.

        This method is called by the game loop to update the state of the game object. Subclasses should override this method to implement the desired behavior.
        
        """
        pass

    @abstractmethod
    def draw(self, screen):
        """Draw the game object to the screen.
        
        :param screen: The surface to draw the game object on.
        :type screen: pygame.Surface
        """
        pass
