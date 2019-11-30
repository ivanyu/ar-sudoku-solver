# -*- coding: utf-8 -*-
from collections import namedtuple


Corners = namedtuple('Corners', 'top_left top_right bottom_right bottom_left')

Field = namedtuple('Field', 'image side margin')

BoundingBox = namedtuple('BoundingBox', 'x y w h')

DISPLAY = True
