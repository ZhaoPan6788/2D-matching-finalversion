#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from geometry import *

import numpy as np
import matplotlib.pyplot as plt

# 材料定义
# 材料定义
vacuum = Material(material_type_vacuum, 'vacuum', {})
electrode_top = Material(material_type_metal, 'top electrode', {})
electrode_bottom = Material(material_type_metal, 'bottom electrode', {})

metal_trg = Material(material_type_metal, 'metal trg', {})
metal_1 = Material(material_type_metal, 'metal 1', {})
metal_2 = Material(material_type_metal, 'metal 2', {})
metal_3 = Material(material_type_metal, 'metal 3', {})
metal_4 = Material(material_type_metal, 'metal 4', {})
metal_circle = Material(material_type_metal, 'metal circle', {})

dielectric_trg = Material(material_type_dielectric, 'diel trg', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_1 = Material(material_type_dielectric, 'diel 1', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_2 = Material(material_type_dielectric, 'diel 2', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_3 = Material(material_type_dielectric, 'diel 3', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_4 = Material(material_type_dielectric, 'diel 4', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})
dielectric_circle = Material(material_type_dielectric, 'diel circle', {
    'permittivity': 12.0, 'permeability': 1.0, 'conductivity': 1.0})

# 几何定义
higth = 32
width = 128

diel_start_h = 5
diel_start_w = 10

metal_start_h = 20
metal_start_w = 10


Nz = higth + 1
Nr = width + 1
geom = Geometry(Nz, Nr)
geom.add_material([vacuum, electrode_top, electrode_bottom,
                  metal_trg, metal_1, metal_2, metal_3, metal_4, metal_circle,
                  dielectric_trg, dielectric_1, dielectric_2, dielectric_3, dielectric_4, dielectric_circle])

geom.add_rectangles(Rectangle(0, 0, higth, width), vacuum.name)
geom.add_rectangles(Rectangle(0, 0, 1, width), electrode_bottom.name)
geom.add_rectangles(Rectangle(higth - 1, 0, 1, width), electrode_top.name)

geom.add_triangles(
    Triangle([(diel_start_h, 0), (diel_start_h, diel_start_w), (diel_start_h+6, 0)]), dielectric_trg.name)
geom.add_triangles(
    Triangle([(metal_start_h, 0), (metal_start_h, metal_start_w), (metal_start_h+6, 0)]), metal_trg.name)

geom.add_rectangles(Rectangle(diel_start_h, 20, 6, 10), dielectric_1.name)
geom.add_rectangles(Rectangle(diel_start_h, 20+5, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(diel_start_h, 40, 6, 10), dielectric_2.name)
geom.add_rectangles(Rectangle(diel_start_h, 40, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(diel_start_h, 60, 6, 10), dielectric_3.name)
geom.add_rectangles(Rectangle(diel_start_h+3, 60+5, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(diel_start_h, 80, 6, 10), dielectric_4.name)
geom.add_rectangles(Rectangle(diel_start_h+3, 80, 3, 5), vacuum.name)

geom.add_rectangles(Rectangle(metal_start_h, 20, 6, 10), metal_1.name)
geom.add_rectangles(Rectangle(metal_start_h, 20+5, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(metal_start_h, 40, 6, 10), metal_2.name)
geom.add_rectangles(Rectangle(metal_start_h, 40, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(metal_start_h, 60, 6, 10), metal_3.name)
geom.add_rectangles(Rectangle(metal_start_h+3, 60+5, 3, 5), vacuum.name)
geom.add_rectangles(Rectangle(metal_start_h, 80, 6, 10), metal_4.name)
geom.add_rectangles(Rectangle(metal_start_h+3, 80, 3, 5), vacuum.name)

geom.add_ellipses(Ellipse((diel_start_h+3, 110), 6, 10),
                  dielectric_circle.name)
geom.add_ellipses(Ellipse((metal_start_h+3, 110), 6, 10), metal_circle.name)


geom.update_edge_type()
geom.update_point_type()
geom.update_epsilon()
geom.update_volume()

geom.save('../../input/')
