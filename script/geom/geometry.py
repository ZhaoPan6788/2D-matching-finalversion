#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from operator import le
import os
import sys
from turtle import left

import h5py
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, Tuple, Union, Dict
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as patches

material_type_vacuum = 0
material_type_metal = 1
material_type_dielectric = 2

point_type_dielectric_vacuum_boundary = -2
point_type_dielectric_inner = -1
point_type_vacuum = 0
point_type_metal = 1

# 从坐标值小的方向指向值大的方向
edge_type_vacuum_to_metal = 0
edge_type_vacuum_to_dielectric = 1
edge_type_metal_to_vacuum = 2
edge_type_dielectric_to_vacuum = 3
edge_type_other = 4


class Rectangle:
    def __init__(self, bottom: int, left: int, higth: int, width: int) -> None:
        self.left = left
        self.bottom = bottom
        self.width = width
        self.higth = higth
        self.check()

    def check(self) -> None:
        if self.left >= 0 and self.bottom >= 0 and self.width > 0 and self.higth > 0:
           pass
        else:
            self.left = 0
            self.bottom = 0
            self.width = 0
            self.higth = 0
            print('Rectangle parameter is invalid.')


class Triangle:
    def __init__(self, corner: list) -> None:
        self.corner = corner
        self.check()

    def check(self) -> None:
        if len(self.corner) == 3:
            for ic in self.corner:
                if len(ic) == 2 and ic[0] >= 0 and ic[1] >= 0:
                    pass
                else:
                    print('Triangle parameter is invalid.')
                    return None
        else:
            print('Triangle parameter is invalid.')
            return None


class Ellipse:
    def __init__(self, center: list, a: int, b: int) -> None:
        self.center = center
        self.a = a
        self.b = b
        self.check()

    def check(self) -> None:
        if len(self.center) == 2 and self.a > 0 and self.b > 0:
            pass
        else:
            print('Ellipse parameter is invalid.')
            return None

class Material:
    def __init__(self, material_type: int, material_name: str, prop_dict: dict) -> None:
        self.type = material_type
        self.name = material_name
        self.prop = prop_dict

    def to_dict(self) -> dict:
        dict = {}
        dict['type'] = self.type
        dict['name'] = self.name
        dict.update(self.prop)

        return dict

    def from_dict(self, dict: dict) -> None:
        self.type = dict['type']
        self.name = dict['name']
        self.prop = dict.copy()
        del self.prop['type']
        del self.prop['name']



# Define a geometry class with an uniform regular mesh, whose different areas can be set to different materials
class Geometry:
    def __init__(self, Nz: int, Nr: int) -> None:
        self.Nz = Nz
        self.Nr = Nr
        self.shape = (self.Nz, self.Nr)

        self.materials = []
        self.cell_material = np.zeros((self.Nz-1, self.Nr-1), dtype=int)        # different materials
        self.point_type = np.zeros((self.Nz, self.Nr), dtype=int)
        self.hedge_type = np.zeros((self.Nz, self.Nr-1), dtype=int)
        self.vedge_type = np.zeros((self.Nz-1, self.Nr), dtype=int)

        self.cell_epsilon_r = np.zeros((self.Nz-1, self.Nr-1), dtype=float)
        self.point_volume = np.zeros((self.Nz, self.Nr), dtype=float)

        self.cell_load = np.zeros((self.Nz-1, self.Nr-1), dtype=int)

    def add_material(self, mate: Union[list, Material]) -> None:
        if isinstance(mate, Material):
            names = [self.materials[i].name for i in range(len(self.materials))]
            if mate.name not in names:
                self.materials.append(mate)

        elif isinstance(mate, list):
            names = [self.materials[i].name for i in range(len(self.materials))]
            for i in range(len(mate)):
                if mate[i].name not in names:
                    self.materials.append(mate[i])

        else:
            print('Invalid input parameter type.')

    def add_rectangles(self, rec: Union[list, Rectangle], material_name: str) -> None:
        material_index = -1
        for i, material in enumerate(self.materials):
            if material.name == material_name:
                material_index = i
                break

        if material_index == -1:
            print('Material not found.')
            return

        if isinstance(rec, Rectangle):
            rec_list = [rec]
        elif isinstance(rec, list):
            rec_list = rec
        else:
            print('Invalid input parameter type.')
            return

        for rectangle in rec_list:
            z_start = rectangle.bottom
            z_end = rectangle.bottom + rectangle.higth
            r_start = rectangle.left
            r_end = rectangle.left + rectangle.width

            self.cell_material[z_start:z_end, r_start:r_end] = material_index

    def add_triangles(self, trg: Union[list, Triangle], material_name: str) -> None:
        material_index = -1
        for i, material in enumerate(self.materials):
            if material.name == material_name:
                material_index = i
                break

        if material_index == -1:
            print('Material not found.')
            return

        if isinstance(trg, Triangle):
            trg_list = [trg]
        elif isinstance(trg, list):
            trg_list = trg
        else:
            print('Invalid input parameter type.')
            return

        for triangle in trg_list:
            A = triangle.corner[0]
            B = triangle.corner[1]
            C = triangle.corner[2]
            AB = (B[0]-A[0], B[1]-A[1])
            BC = (C[0]-B[0], C[1]-B[1])
            CA = (A[0]-C[0], A[1]-C[1])

            z_start = min([A[0], B[0], C[0]])
            z_end = max([A[0], B[0], C[0]])
            r_start = min([A[1], B[1], C[1]])
            r_end = max([A[1], B[1], C[1]])

            for i in range(z_start, z_end):
                for j in range(r_start, r_end):
                    P = (i+0.5, j+0.5)
                    AP = (P[0]-A[0], P[1]-A[1])
                    BP = (P[0]-B[0], P[1]-B[1])
                    CP = (P[0]-C[0], P[1]-C[1])

                    if AP[0]*AB[1]-AP[1]*AB[0] >= 0 and BP[0]*BC[1]-BP[1]*BC[0] >= 0 and CP[0]*CA[1]-CP[1]*CA[0] >= 0:
                        self.cell_material[i, j] = material_index

    def add_ellipses(self, elp: Union[list, Ellipse], material_name: str) -> None:
        material_index = -1
        for i, material in enumerate(self.materials):
            if material.name == material_name:
                material_index = i
                break

        if material_index == -1:
            print('Material not found.')
            return

        if isinstance(elp, Ellipse):
            elp_list = [elp]
        elif isinstance(elp, list):
            elp_list = elp
        else:
            print('Invalid input parameter type.')
            return

        for ellipse in elp_list:
            z0 = float(ellipse.center[0])
            r0 = float(ellipse.center[1])
            a = float(ellipse.a)
            b = float(ellipse.b)

            z_start = ellipse.center[0] - ellipse.a
            z_end = ellipse.center[0] + ellipse.a
            r_start = ellipse.center[1] - ellipse.b
            r_end = ellipse.center[1] + ellipse.b

            for i in range(z_start, z_end):
                for j in range(r_start, r_end):
                    P = (i+0.5, j+0.5)

                    if ((P[0] - z0)/a)**2 + ((P[1] - r0)/b)**2 < 1.0:
                        self.cell_material[i, j] = material_index

    def update_point_type(self):
        for z in range(self.Nz):
            for r in range(self.Nr):
                if 0 == z and 0 == r:
                    mate_type_tmp = self.materials[self.cell_material[z, r]].type
                    if mate_type_tmp == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif mate_type_tmp == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum
                    
                    if mate_type_tmp == material_type_metal:
                        self.point_type[z, r] = point_type_metal

                elif 0 == z and r < self.Nr-1:
                    left_mate = self.materials[self.cell_material[z, r-1]].type
                    right_mate = self.materials[self.cell_material[z, r]].type

                    if left_mate == material_type_metal or right_mate == material_type_metal:
                        self.point_type[z, r] = point_type_metal
                    elif left_mate == material_type_dielectric and right_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif left_mate == material_type_vacuum and right_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                elif 0 == z and r == self.Nr-1:
                    mate_type_tmp = self.materials[self.cell_material[z, r-1]].type
                    if mate_type_tmp == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif mate_type_tmp == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                    if mate_type_tmp == material_type_metal:
                        self.point_type[z, r] = point_type_metal

                elif self.Nz-1 == z and 0 == r:
                    mate_type_tmp = self.materials[self.cell_material[z-1, r]].type
                    if mate_type_tmp == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif mate_type_tmp == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                    if mate_type_tmp == material_type_metal:
                        self.point_type[z, r] = point_type_metal

                elif self.Nz-1 == z and r < self.Nr-1:
                    left_mate = self.materials[self.cell_material[z-1, r-1]].type
                    right_mate = self.materials[self.cell_material[z-1, r]].type

                    if left_mate == material_type_metal or right_mate == material_type_metal:
                        self.point_type[z, r] = point_type_metal
                    elif left_mate == material_type_dielectric and right_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif left_mate == material_type_vacuum and right_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                elif self.Nz-1 == z and r == self.Nr-1:
                    mate_type_tmp = self.materials[self.cell_material[z-1, r-1]].type
                    if mate_type_tmp == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif mate_type_tmp == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                    if mate_type_tmp == material_type_metal:
                        self.point_type[z, r] = point_type_metal

                elif 0 == r:
                    bottom_mate = self.materials[self.cell_material[z-1, r]].type
                    top_mate = self.materials[self.cell_material[z, r]].type

                    if bottom_mate == material_type_metal or top_mate == material_type_metal:
                        self.point_type[z, r] = point_type_metal
                    elif bottom_mate == material_type_dielectric and top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif bottom_mate == material_type_vacuum and top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                elif self.Nr-1 == r:
                    bottom_mate = self.materials[self.cell_material[z-1, r-1]].type
                    top_mate = self.materials[self.cell_material[z, r-1]].type

                    if bottom_mate == material_type_metal or top_mate == material_type_metal:
                        self.point_type[z, r] = point_type_metal
                    elif bottom_mate == material_type_dielectric and top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner
                    elif bottom_mate == material_type_vacuum and top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum

                else:
                    left_bottom_mate = self.materials[self.cell_material[z-1, r-1]].type
                    left_top_mate = self.materials[self.cell_material[z, r-1]].type
                    right_bottom_mate = self.materials[self.cell_material[z-1, r]].type
                    right_top_mate = self.materials[self.cell_material[z, r]].type

                    if left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_vacuum
                    
                    elif left_bottom_mate == material_type_metal or \
                            left_top_mate == material_type_metal or \
                            right_bottom_mate == material_type_metal or \
                            right_top_mate == material_type_metal:
                        self.point_type[z, r] = point_type_metal

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_inner

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_dielectric and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_vacuum:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_dielectric and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_vacuum and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

                    elif left_bottom_mate == material_type_vacuum and \
                            left_top_mate == material_type_vacuum and \
                            right_bottom_mate == material_type_dielectric and \
                            right_top_mate == material_type_dielectric:
                        self.point_type[z, r] = point_type_dielectric_vacuum_boundary

    def update_edge_type(self):
        # Loop through horizontal edges
        self.hedge_type[:, :] = edge_type_other
        for z in range(1, self.Nz-1):
            for r in range(self.Nr - 1):
                top_material = self.materials[self.cell_material[z, r]].type
                bottom_material = self.materials[self.cell_material[z-1, r]].type

                if top_material != bottom_material:
                    if bottom_material == material_type_vacuum and top_material == material_type_metal:
                        self.hedge_type[z, r] = edge_type_vacuum_to_metal
                    elif bottom_material == material_type_vacuum and top_material == material_type_dielectric:
                        self.hedge_type[z, r] = edge_type_vacuum_to_dielectric
                    elif bottom_material == material_type_metal and top_material == material_type_vacuum:
                        self.hedge_type[z, r] = edge_type_metal_to_vacuum
                    elif bottom_material == material_type_dielectric and top_material == material_type_vacuum:
                        self.hedge_type[z, r] = edge_type_dielectric_to_vacuum

        # Loop through vertical edges
        self.vedge_type[:, :] = edge_type_other
        for z in range(self.Nz-1):
            for r in range(1, self.Nr - 1):
                right_material = self.materials[self.cell_material[z, r]].type
                left_material = self.materials[self.cell_material[z, r-1]].type

                if right_material != left_material:
                    if left_material == material_type_vacuum and right_material == material_type_metal:
                        self.vedge_type[z, r] = edge_type_vacuum_to_metal
                    elif left_material == material_type_vacuum and right_material == material_type_dielectric:
                        self.vedge_type[z, r] = edge_type_vacuum_to_dielectric
                    elif left_material == material_type_metal and right_material == material_type_vacuum:
                        self.vedge_type[z, r] = edge_type_metal_to_vacuum
                    elif left_material == material_type_dielectric and right_material == material_type_vacuum:
                        self.vedge_type[z, r] = edge_type_dielectric_to_vacuum

    def update_epsilon(self):
        self.cell_epsilon_r = np.ones((self.Nz-1, self.Nr-1), dtype=float)

        for z in range(self.Nz-1):
            for r in range(self.Nr-1):
                if self.materials[self.cell_material[z, r]].type == material_type_dielectric:
                    self.cell_epsilon_r[z, r] = self.materials[self.cell_material[z, r]].prop['permittivity']
 
    def update_volume(self):
        self.point_volume = np.zeros((self.Nz, self.Nr), dtype=float)

        # 四个角点
        if self.materials[self.cell_material[0, 0]].type == material_type_vacuum:
            self.point_volume[0, 0] = 0.5
        
        if self.materials[self.cell_material[self.Nz-2, 0]].type == material_type_vacuum:
            self.point_volume[self.Nz-1, 0] = 0.5

        if self.materials[self.cell_material[0, self.Nr-2]].type == material_type_vacuum:
            self.point_volume[0, self.Nr-1] = 0.5

        if self.materials[self.cell_material[self.Nz-2, self.Nr-2]].type == material_type_vacuum:
            self.point_volume[self.Nz-1, self.Nr-1] = 0.5

        # 边界点
        r = 0
        for z in range(1, self.Nz-1):
            s1 = 0
            s2 = 0
            if self.materials[self.cell_material[z-1, r]].type == material_type_vacuum:
                s1 = 0.5
            
            if self.materials[self.cell_material[z, r]].type == material_type_vacuum:
                s2 = 0.5

            self.point_volume[z, r] = s1 + s2

        r = self.Nr-1
        for z in range(1, self.Nz-1):
            s1 = 0
            s2 = 0
            if self.materials[self.cell_material[z-1, r-1]].type == material_type_vacuum:
                s1 = 0.5
            
            if self.materials[self.cell_material[z, r-1]].type == material_type_vacuum:
                s2 = 0.5

            self.point_volume[z, r] = s1 + s2

        z = 0
        for r in range(1, self.Nr-1):
            s1 = 0
            s2 = 0
            if self.materials[self.cell_material[z, r-1]].type == material_type_vacuum:
                s1 = 0.25 * (1 - 1/4/r)
            
            if self.materials[self.cell_material[z, r]].type == material_type_vacuum:
                s2 = 0.25 * (1 + 1/4/r)

            self.point_volume[z, r] = s1 + s2

        z = self.Nz-1
        for r in range(1, self.Nr-1):
            s1 = 0
            s2 = 0
            if self.materials[self.cell_material[z-1, r-1]].type == material_type_vacuum:
                s1 = 0.25 * (1 - 1/4/r)

            if self.materials[self.cell_material[z-1, r]].type == material_type_vacuum:
                s2 = 0.25 * (1 + 1/4/r)

            self.point_volume[z, r] = s1 + s2


        for z in range(1, self.Nz-1):
            for r in range(1, self.Nr-1):
                s1 = 0
                s2 = 0
                s3 = 0
                s4 = 0

                if self.materials[self.cell_material[z-1, r-1]].type == material_type_vacuum:
                    s1 = 0.25 * (1 - 1/4/r)

                if self.materials[self.cell_material[z-1, r]].type == material_type_vacuum:
                    s2 = 0.25 * (1 + 1/4/r)

                if self.materials[self.cell_material[z, r-1]].type == material_type_vacuum:
                    s3 = 0.25 * (1 - 1/4/r)

                if self.materials[self.cell_material[z, r]].type == material_type_vacuum:
                    s4 = 0.25 * (1 + 1/4/r)

                self.point_volume[z, r] = s1 + s2 + s3 + s4

        self.update_load()

    def update_load(self, func=None):
        self.cell_load = np.zeros((self.Nz-1, self.Nr-1), dtype=int)

        if func != None:
            for z in range(self.Nz-1):
                for r in range(self.Nr-1):
                    self.cell_load[z, r] = func(z, r)

        else: 
            for z in range(self.Nz-1):
                for r in range(self.Nr-1):
                    if self.materials[self.cell_material[z, r]].type == material_type_vacuum:
                        self.cell_load[z, r] = 1

    def show(self, material_colors: Optional[Dict[str, Tuple[float, float, float]]] = None, savepath: str = None) -> None:
        plt.figure(figsize=(12, 8))

        if material_colors is None:
            # Default colors if not specified
            material_colors = {material.name: color for material, color in zip(
                self.materials, plt.cm.viridis(np.linspace(0, 1, len(self.materials))))}

        color_map = np.zeros((self.Nz - 1, self.Nr - 1, 3))
        for i, material in enumerate(self.materials):
            mask = (self.cell_material == i)
            for j in range(3):
                color_map[mask, j] = material_colors[material.name][j]

        plt.imshow(color_map, origin='lower', extent=(
            0, self.Nr - 1, 0, self.Nz - 1), aspect='auto')

        # Create legend for materials
        legend_elements = [
            Patch(facecolor=material_colors[material.name],
                  edgecolor='k', label=material.name)
            for material in self.materials
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=[1.22, 1])

        plt.title('Material Distribution')

        if savepath:
            plt.savefig(os.path.join(savepath), dpi=300,
                        pad_inches=0.2, bbox_inches='tight')
        else:
            plt.show()

    def show_grid(self, savepath: str = None, is_create_fig: bool = True, color='k', isvert=False) -> None:
        if is_create_fig:
            plt.figure(figsize=(12, 8))

        if not isvert:
            # Draw horizontal grid lines
            for z in range(1, self.Nz - 1):
                for r in range(self.Nr - 1):
                    if self.cell_material[z, r] != self.cell_material[z - 1, r]:
                        plt.hlines(y=z, xmin=r, xmax=r+1,
                                color=color, linewidth=0.5)

            # Draw vertical grid lines
            for r in range(1, self.Nr - 1):
                for z in range(self.Nz - 1):
                    if self.cell_material[z, r] != self.cell_material[z, r - 1]:
                        plt.vlines(x=r, ymin=z, ymax=z+1,
                                color=color, linewidth=0.5)

        else:
            # Draw horizontal grid lines
            for z in range(1, self.Nz - 1):
                for r in range(self.Nr - 1):
                    if self.cell_material[z, r] != self.cell_material[z - 1, r]:
                        plt.vlines(x=z, ymin=r, ymax=r+1,
                                color=color, linewidth=0.5)

            # Draw vertical grid lines
            for r in range(1, self.Nr - 1):
                for z in range(self.Nz - 1):
                    if self.cell_material[z, r] != self.cell_material[z, r - 1]:
                        plt.hlines(y=r, xmin=z, xmax=z+1,
                                color=color, linewidth=0.5)

        if is_create_fig:
            if not isvert:
                plt.xlim(0, self.Nr - 1)
                plt.ylim(0, self.Nz - 1)
                plt.xlabel('R-axis')
                plt.ylabel('Z-axis')
            else:
                plt.xlim(0, self.Nz - 1)
                plt.ylim(0, self.Nr - 1)
                plt.xlabel('Z-axis')
                plt.ylabel('R-axis')

            plt.title('Material Boundary Grid')

            if savepath:
                plt.savefig(os.path.join(savepath), dpi=300,
                            pad_inches=0.2, bbox_inches='tight')
            else:
                plt.show()

    def show_edge(self, savepath: str = None) -> None:
        plt.figure(figsize=(12, 8))
    
        edge_colors = {
            edge_type_vacuum_to_metal: 'r',
            edge_type_vacuum_to_dielectric: 'b',
            edge_type_metal_to_vacuum: 'g',
            edge_type_dielectric_to_vacuum: 'y',
            edge_type_other: 'k'
        }

        # Plot horizontal edges
        for z in range(1, self.Nz-1):
            for r in range(self.Nr - 1):
                if self.hedge_type[z, r] != edge_type_other:
                    plt.hlines(y=z, xmin=r, xmax=r+1, color=edge_colors[self.hedge_type[z, r]], linewidth=2)

        # Plot vertical edges
        for z in range(self.Nz-1):
            for r in range(1, self.Nr - 1):
                if self.vedge_type[z, r] != edge_type_other:
                    plt.vlines(x=r, ymin=z, ymax=z+1, color=edge_colors[self.vedge_type[z, r]], linewidth=2)

        plt.xlim(0, self.Nr - 1)
        plt.ylim(0, self.Nz - 1)
        plt.xlabel('R-axis')
        plt.ylabel('Z-axis')
        plt.title('Material Edge Grid')

        if savepath:
            plt.savefig(os.path.join(savepath), dpi=300, pad_inches=0.2, bbox_inches='tight')
        else:
            plt.show()

    def show_point(self, savepath: str = None) -> None:
        plt.figure(figsize=(12, 8))
        cmap = plt.cm.viridis

        r_values, z_values = np.mgrid[0:self.Nr:, 0:self.Nz]
        r = np.array(r_values.flatten())
        z = np.array(z_values.flatten())

        plt.scatter(r, z, c=self.point_type.T.flatten(), cmap=cmap, marker='s')

        plt.xlabel('R axis')
        plt.ylabel('Z axis')
        plt.title('Point Type Distribution')
        plt.colorbar(ticks=range(-2, 13), label='Point Types')
        
        if savepath:
            plt.savefig(os.path.join(savepath), dpi=300, pad_inches=0.2, bbox_inches='tight')
        else:
            plt.show()

    def save(self, path: str, name: str = None) -> None:
        filename = 'geom'
        if name:
            filename = name

        with h5py.File(os.path.join(path, filename+'.h5'), 'w') as f:
            # Save geometry attributes
            f.attrs['Nz'] = self.Nz
            f.attrs['Nr'] = self.Nr

            # Save cell_material, point_type, hedge_type, and vedge_type arrays
            f.create_dataset('cell_material', data=self.cell_material.T, dtype=np.int32)
            f.create_dataset('point_type', data=self.point_type.T, dtype=np.int32)
            f.create_dataset('hedge_type', data=self.hedge_type.T, dtype=np.int32)
            f.create_dataset('vedge_type', data=self.vedge_type.T, dtype=np.int32)
            f.create_dataset('epsilon_r',  data=self.cell_epsilon_r.T, dtype=np.float64)
            f.create_dataset('volume',  data=self.point_volume.T, dtype=np.float64)
            f.create_dataset('load',  data=self.cell_load.T, dtype=np.int32)

        with open(os.path.join(path, filename+'.json'), 'w') as f:
            dict = {}
            mates = []
            for i in range(len(self.materials)):
                mates.append(self.materials[i].to_dict())

            dict['material_amount'] = len(self.materials)
            dict['material_type'] = {
                'material_type_vacuum': material_type_vacuum,
                'material_type_metal': material_type_metal,
                'material_type_dielectric': material_type_dielectric,
            }
            dict['point_type'] = {
                'point_type_vacuum': point_type_vacuum,
                'point_type_metal': point_type_metal,
                'point_type_dielectric_inner': point_type_dielectric_inner,
                'point_type_dielectric_vacuum_boundary': point_type_dielectric_vacuum_boundary,
            }
            dict['edge_type'] = {
                'edge_type_vacuum_to_metal': edge_type_vacuum_to_metal,
                'edge_type_vacuum_to_dielectric': edge_type_vacuum_to_dielectric,
                'edge_type_metal_to_vacuum': edge_type_metal_to_vacuum,
                'edge_type_dielectric_to_vacuum': edge_type_dielectric_to_vacuum,
            }

            dict['materials'] = mates

            json.dump(dict, f, indent=4)

    def load(self, path: str, name: str = None) -> None:
        filename = 'geom'
        if name:
            filename = name

        with h5py.File(os.path.join(path, filename+'.h5'), 'r') as f:
            # Load geometry attributes
            self.Nz = f.attrs['Nz']
            self.Nr = f.attrs['Nr']

            # Load cell_material, point_type, hedge_type, and vedge_type arrays
            self.cell_material = np.array(f['cell_material']).T
            self.point_type = np.array(f['point_type']).T
            self.hedge_type = np.array(f['hedge_type']).T
            self.vedge_type = np.array(f['vedge_type']).T
            self.cell_epsilon_r = np.array(f['epsilon_r']).T
            self.point_volume = np.array(f['volume']).T
            self.cell_load = np.array(f['load']).T

        with open(os.path.join(path, filename+'.json'), 'r') as f:
            dict = json.load(f)
            num = dict['material_amount']
            self.materials.clear()
            mates = dict['materials']

            for i in range(num):
                tmp = Material(material_type_vacuum, 'tmp', {})
                tmp.from_dict(mates[i])
                print(tmp.name, tmp.type, tmp.prop)
                self.materials.append(tmp)
