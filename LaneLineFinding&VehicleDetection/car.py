#!/usr/bin/python
"""

"""
class Car(object):
    """Car object

    Attributes
    ----------
    loc: tuple, 2x1
        Coordinates (x, y) of the center of the car.
    shape: tuple, 2x1
        Shape (width, height) of the car.
    name: string
        Name of the car.
    color: string
        Color of the car.
    type: string
        Type of the car.
    """
    def __init__(self, *args, **kwargs):
        """Initialization"""
        self.loc = None
        self.shape = None

        self.name = None
        self.color = None
        self.type = None

        for key in kwargs.keys():
            if key == 'loc':
                self.loc = tuple(kwargs[key])
            elif key == 'shape':
                self.shape = tuple(kwargs[key])
            elif key == 'name':
                self.name = kwargs[key]
            elif key == 'color':
                self.color = kwargs[key]
            elif key == 'type':
                self.type = kwargs[key]
            else:
                pass

        if not isinstance(self.loc, tuple):
            raise TypeError()
        elif len(self.loc) != 2:
            raise ValueError("Length of loc must be 2!")

        if not isinstance(self.shape, tuple):
            raise TypeError()
        elif len(self.shape) != 2:
            raise ValueError("Length of shape must be 2!")