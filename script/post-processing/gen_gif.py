#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import os
import re
import math
import h5py

import sys
import os
import re
import shutil

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from base.common import get_domain_info, draw_domain
from geom.geometry import *
from base.common import *

import numpy as np
import matplotlib.pyplot as plt

import h5py
import imageio.v2 as imageio
import multiprocessing as mp
from functools import partial


def compose_gif(filePath, gifName, duration=20):
    dirPath = os.path.abspath(filePath)
    fileList = os.listdir(dirPath)
    fileList.sort(reverse=False)

    if len(fileList) > 0:
        tmpDir = list()

        for i in range(len(fileList)):
            if len(re.findall(re.escape(gifName)+'_\d*[.]png', fileList[i])) > 0:
                tmpDir.append(fileList[i])

        if len(tmpDir) > 0:
            fileList = tmpDir
        else:
            print('The number of directories that meet the rule is 0.')
            exit(-1)

    gif_images = []
    tmp = imageio.imread(os.path.join(dirPath, fileList[0]))
    tmp_shape = tmp.shape
    for file in fileList:
        path = os.path.join(dirPath, file)
        tmp = imageio.imread(path)
        if (tmp.shape == tmp_shape):
            gif_images.append(tmp)

    imageio.mimsave(gifName+'.gif', gif_images, duration=duration)


def initDraw(drawDirName='.draw'):
    drawPath = os.path.join(os.getcwd(), drawDirName)
    if os.path.exists(drawPath):
        shutil.rmtree(drawPath)

    os.mkdir(drawPath)

    return drawPath


def create_fig(var_name, h5files, max_num=100, islimit=False, lev=20, du=10):
    h5num = len(h5files)
    num = max_num
    if num > h5num:
        num = h5num

    dmin = np.min(np.array(h5files[0][var_name]))
    dmax = np.max(np.array(h5files[0][var_name]))

    zpos = np.arange(g.Nz)
    rpos = np.arange(g.Nr)
    [rr, zz] = np.meshgrid(rpos, zpos)

    drawpath = initDraw(drawDirName=var_name)
    for j in range(num):
        i = int(j * h5num * 1.0 / num)
        rho = np.array(h5files[i][var_name]).T
        if islimit:
            tmp_min = np.min(rho)
            tmp_max = np.max(rho)
            if dmin > tmp_min:
                dmin = tmp_min

            if dmax < tmp_max:
                dmax = tmp_max
        else:
            dmin = np.min(rho)
            dmax = np.max(rho)

        font, size, sm, lims = contour(
            rr, zz, rho, lev=lev, bar=False, z_min=dmin, z_max=dmax)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)

        cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                          boundaries=lims)
        cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
        cb.ax.tick_params(length=6, width=1.5)
        cb.outline.set_linewidth(1.5)
        cb.set_label(var_name, fontdict=font)  # 设置colorbar的标签字体及其大小

        plt.grid(False)
        plt.xlabel('R', fontdict=font)
        plt.ylabel('Z', fontdict=font)
        plt.title('NO.{0}'.format(i))

        g.show_grid(is_create_fig=False, color='w', isvert=False)
        # draw_domain(domain_info[0], is_create_fig=False, color='w')

        plt.savefig(os.path.join(drawpath, var_name +
                    '_{0:0>5d}.png'.format(i)), dpi=100, pad_inches=0.2)

    compose_gif(drawpath, var_name, duration=du)



# info
domain_info = get_domain_info('../data/B1/check/domain.dat')
g = Geometry(1, 1)
g.load('../data/B1/input/')


filepath = '../data/B1/diag/'
filestr  = 'DiagParticleField2D'

# task dirs
dirPath = os.path.abspath(filepath)
subDirs = os.listdir(dirPath)
subDirs.sort(reverse = False)

if len(subDirs) > 0:
    tmpDir = list()

    for i in range(len(subDirs)):
        if len(re.findall(filestr+'_\d*[.]h5', subDirs[i])) > 0:
            tmpDir.append(subDirs[i])

    if len(tmpDir) > 0:
        subDirs = tmpDir
    else:
        print('The number of directories that meet the rule is 0.')
        exit(-1)

# print(subDirs)

# read hdf5
h5files = []
for i in range(len(subDirs)):
    h5files.append(h5py.File(os.path.join(dirPath, subDirs[i]),'r'))


# key
key_list = list()
if len(h5files) > 0:
    i = 0
    print('The keys in h5 files.')
    for ikey in h5files[0].keys():
        print('{0:<3} : {1:<}'.format(i, h5files[0][ikey].name))
        key_list.append(h5files[0][ikey].name)
        i = i + 1
else:
    print('Not found h5 files.')
    exit(-1)

print('')

# draw all
def create_fig_with_index(index):
    try:
        varname = key_list[index][1:]
        create_fig(varname, h5files, du=0.02)
    except ValueError:
        print('Invalid draw.')

with mp.Pool(processes=2) as pool:
    pool.map(create_fig_with_index, range(len(key_list)))


# for i in range(len(key_list)):
#     try:
#         varname = key_list[i][1:]
#         create_fig(varname, h5files, du=0.02)
#     except ValueError:
#         print('Invalid draw.')

# create_fig('RhoOne-Electron', h5files, du=0.02)
# create_fig('EnergyOne-Electron', h5files, du=0.02)
# create_fig('Phi', h5files, du=0.02)
