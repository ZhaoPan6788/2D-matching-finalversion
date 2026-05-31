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
k = 1
higth = 80 * k
width = 80 * k
electrode_width = 38 * k
electrode_higth = 30 * k
wall_thickness = 1 * k
dielectric_thickness = 1 * k

Nz = higth + 1
Nr = width + 1
geom = Geometry(Nz, Nr)
geom.add_material([vacuum, electrode_top, electrode_bottom,
                  wall, dielectric_1, dielectric_2])

geom.add_rectangles(Rectangle(0, 0, higth, width), 'wall')
geom.add_rectangles(Rectangle(0, 0, electrode_higth,
                    electrode_width), 'bottom electrode')
geom.add_rectangles(Rectangle(higth - electrode_higth, 0,
                    electrode_higth, electrode_width), 'top electrode')
geom.add_rectangles(Rectangle(0, electrode_width,
                    electrode_higth, dielectric_thickness), 'dielectric 1')
geom.add_rectangles(Rectangle(higth - electrode_higth, electrode_width,
                    electrode_higth, dielectric_thickness), 'dielectric 2')
geom.add_rectangles(Rectangle(electrode_higth, 0, higth - electrode_higth*2,
                    electrode_width + dielectric_thickness + wall_thickness), 'vacuum')
geom.add_rectangles(Rectangle(wall_thickness, electrode_width + dielectric_thickness + wall_thickness,
                    higth - wall_thickness*2, width - electrode_width - dielectric_thickness - wall_thickness*2), 'vacuum')

geom.update_edge_type()
geom.update_point_type()
geom.update_epsilon()
geom.update_volume()

geom.save('../../input/')
