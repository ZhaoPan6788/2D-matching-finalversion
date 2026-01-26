#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from geometry import *

import numpy as np
import matplotlib.pyplot as plt

# 材料定义
vacuum = Material(material_type_vacuum, 'vacuum', {})
electrode_top = Material(material_type_metal, 'top electrode', {})
electrode_bottom = Material(material_type_metal, 'bottom electrode', {})
wall = Material(material_type_metal, 'wall', {})
dielectric_1 = Material(material_type_dielectric, 'dielectric 1', {
                        'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_2 = Material(material_type_dielectric, 'dielectric 2', {
                        'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})

# 几何定义
higth = 32
width = 128

Nz = higth + 1
Nr = width + 1
geom = Geometry(Nz, Nr)
geom.add_material([vacuum, electrode_top, electrode_bottom,
                  wall, dielectric_1, dielectric_2])

geom.add_rectangles(Rectangle(0, 0, higth, width), 'vacuum')
geom.add_rectangles(Rectangle(0, 0, 1, width), 'bottom electrode')
geom.add_rectangles(Rectangle(higth - 1, 0, 1, width), 'top electrode')

geom.update_edge_type()
geom.update_point_type()
geom.update_epsilon()
geom.update_volume()

geom.save('../../input/')
