#!/usr/bin/env python3
#-*- coding: utf-8 -*-

#%% env
import sys
import os
import re
import math
import h5py

sys.path.append(os.path.split(os.path.abspath(''))[0])
from base.read_matrix import MatrixDataReader
from geom.geometry import *
from base.common import *

import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fsave_set_save(True)
fsave_set_dpi(300)
fsave_set_pad_inches(0.2)

#%% info
sim_path = '../../run'
filepath = os.path.join(sim_path, 'diag')
filestr = 'DiagParticleField2D'
mean_num = 10
dz = 6.2500E-04
dr = dz

#%% domain
domain_info = get_domain_info(os.path.join(sim_path, 'check/domain.dat'))
g = Geometry(1, 1)
g.load(os.path.join(sim_path, 'input'))
Nz = g.Nz
Nr = g.Nr

plt.figure(figsize=(10, 6))

g.show_grid(is_create_fig=False)
draw_domain(domain_info[0], is_create_fig=False)

plt.xlabel('R-axis',)
plt.ylabel('Z-axis',)
plt.xlim([0, g.Nr-1])
plt.ylim([0, g.Nz-1])

fs('./domain')

#%% task dirs
dirPath = os.path.abspath(filepath)
subDirs = os.listdir(dirPath)
subDirs.sort(reverse=False)

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

print('H5 file number: ', len(subDirs))

#%%
time = np.array([i for i in range(len(subDirs))])
ne_mean = np.zeros((len(subDirs), 1))
ni_mean = np.zeros((len(subDirs), 1))
Te_mean = np.zeros((len(subDirs), 1))
Ti_mean = np.zeros((len(subDirs), 1))
phi_mean = np.zeros((len(subDirs), 1))

ro = 100
for j in range(math.ceil(len(subDirs)/ro)):
    start = j * ro
    stop = (j + 1) * ro
    if stop > len(subDirs):
        stop = len(subDirs)

    h5files = []
    for i in range(start, stop):
        h5files.append(h5py.File(os.path.join(dirPath, subDirs[i]), 'r'))

    for i in range(len(h5files)):
        ne_mean[i+start] = np.mean(h5files[i]['RhoOne-Electron'])
        ni_mean[i+start] = np.mean(h5files[i]['RhoOne-Ar+'])
        Te_mean[i+start] = np.mean(h5files[i]['EnergyOne-Electron'])
        Ti_mean[i+start] = np.mean(h5files[i]['EnergyOne-Ar+'])
        phi_mean[i+start] = np.mean(h5files[i]['Phi'])

    for i in range(len(h5files)):
        h5files[i].close()

#%% density-time
font, size = curvestyle()

plt.plot(time, ne_mean, lw=3, c=colors[0], label='$n_e$')
plt.plot(time, ni_mean, lw=3, c=colors[1], label='$n_i$')

plt.xlabel('Time (RF)', fontdict=font)
plt.ylabel('Density ($m^{-3}$)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./ne-time')

#%% energy-time
font, size = curvestyle()

plt.plot(time, Te_mean, lw=3, c=colors[0], label="$T_e$")
plt.plot(time, Ti_mean, lw=3, c=colors[1], label="$T_i$")

plt.xlabel('Time (RF)', fontdict=font)
plt.ylabel('Temperature (eV)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./Te-time')

#%% bias-time
font, size = curvestyle()

plt.plot(time, phi_mean, lw=3, c=colors[0])

plt.xlabel('Time (RF)', fontdict=font)
plt.ylabel('Voltage (V)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
# plt.legend(prop={'size':20, 'family':'Arial'}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./vol-time')

#%% read hdf5 for steady
h5files = []
for i in range(len(subDirs)-mean_num, len(subDirs)):
    h5files.append(h5py.File(os.path.join(dirPath, subDirs[i]), 'r'))

#%% ne ni
varname = 'RhoOne-Electron'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(
    rr, zz, get_mean_field(h5files, varname, -mean_num).T, lev=20, bar=False, z_min=0)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./ne-2d')


varname = 'RhoOne-Ar+'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False, z_min=0)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./ni-2d')

#%% ne profile
varname = 'RhoOne-Electron'
time_index = -1

rho_2d = h5files[time_index][varname]
zpos = np.arange(Nz)

font, size = curvestyle()

index = 0
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[0], label='r = {}'.format(index))
index = int(Nr / 4)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[1], label='r = {}'.format(index))
index = int(Nr / 4 * 2)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[2], label='r = {}'.format(index))
index = int(Nr / 4 * 3)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[3], label='r = {}'.format(index))

plt.xlabel('Position Z', fontdict=font)
plt.ylabel('Density ($m^{-3}$)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./ne_zpos')


varname = 'RhoOne-Electron'
time_index = -1

rho_2d = h5files[time_index][varname]
rpos = np.arange(Nr)

font, size = curvestyle()

index = int(Nz / 4)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[0], label='z = {}'.format(index))
index = int(Nz / 4 * 2)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[1], label='z = {}'.format(index))
index = int(Nz / 4 * 3)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[2], label='z = {}'.format(index))

plt.xlabel('Position R', fontdict=font)
plt.ylabel('Density ($m^{-3}$)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./ne_rpos')


#%% Te
varname = 'EnergyOne-Electron'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False, z_min=0)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Te-2d')


varname = 'EnergyOne-Ar+'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False, z_min=0)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Ti-2d')


#%% Te profile
varname = 'EnergyOne-Electron'
time_index = -1

rho_2d = h5files[time_index][varname]
zpos = np.arange(Nz)

font, size = curvestyle()

index = 0
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[0], label='r = {}'.format(index))
index = int(Nr / 4)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[1], label='r = {}'.format(index))
index = int(Nr / 4 * 2)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[2], label='r = {}'.format(index))
index = int(Nr / 4 * 3)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[3], label='r = {}'.format(index))

plt.xlabel('Position Z', fontdict=font)
plt.ylabel('Temperature (eV)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./Te_zpos')


varname = 'EnergyOne-Electron'
time_index = -1

rho_2d = h5files[time_index][varname]
rpos = np.arange(Nr)

font, size = curvestyle()

index = int(Nz / 4)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[0], label='z = {}'.format(index))
index = int(Nz / 4 * 2)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[1], label='z = {}'.format(index))
index = int(Nz / 4 * 3)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[2], label='z = {}'.format(index))

plt.xlabel('Position R', fontdict=font)
plt.ylabel('Temperature (eV)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./Te_rpos')


#%% phi
varname = 'Phi'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./phi-2d')


#%% phi profile
varname = 'Phi'
time_index = -1

rho_2d = h5files[time_index][varname]
zpos = np.arange(Nz)

font, size = curvestyle()

index = 0
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[0], label='r = {}'.format(index))
index = int(Nr / 4)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[1], label='r = {}'.format(index))
index = int(Nr / 4 * 2)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[2], label='r = {}'.format(index))
index = int(Nr / 4 * 3)
plt.plot(zpos, rho_2d[index, :], lw=3,
         c=colors[3], label='r = {}'.format(index))

plt.xlabel('Position Z', fontdict=font)
plt.ylabel('Potential (V)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./phi_zpos')


varname = 'Phi'
time_index = -1

rho_2d = h5files[time_index][varname]
rpos = np.arange(Nr)

font, size = curvestyle()

index = int(Nz / 4)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[0], label='z = {}'.format(index))
index = int(Nz / 4 * 2)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[1], label='z = {}'.format(index))
index = int(Nz / 4 * 3)
plt.plot(rpos, rho_2d[:, index], lw=3,
         c=colors[2], label='z = {}'.format(index))

plt.xlabel('Position R', fontdict=font)
plt.ylabel('Potential (V)', fontdict=font)
# plt.xlim([-0.1, 1.1])
# plt.ylim([10**18.7, 10**21.2])
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(prop={'size': 20}, frameon=False)

ax = plt.gca()
# ax.xaxis.set_minor_locator(MultipleLocator(0.1))

fs('./phi_rpos')


#%% E
Ez, Er = calculate_electric_field(
    get_mean_field(h5files, varname, -mean_num), dz, dr)

font, size, sm, lims = contour(rr, zz, Ez.T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Ez-2d')


font, size, sm, lims = contour(rr, zz, Er.T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Er-2d')


#%% Jz
varname = 'JzOne-Electron'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Jze-2d')


varname = 'JzOne-Ar+'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Jzi-2d')


#%% Jr
varname = 'JrOne-Electron'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Jre-2d')


varname = 'JrOne-Ar+'
zpos = np.arange(Nz)
rpos = np.arange(Nr)
[rr, zz] = np.meshgrid(rpos, zpos)
font, size, sm, lims = contour(rr, zz, get_mean_field(
    h5files, varname, -mean_num).T, lev=20, bar=False)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)

cb = plt.colorbar(sm, fraction=0.15, pad=0.1,
                  boundaries=lims)
cb.ax.tick_params(labelsize=size)     # 设置色标刻度字体大小
cb.ax.tick_params(length=6, width=1.5)
cb.outline.set_linewidth(1.5)
g.show_grid(is_create_fig=False, color='w')
# draw_domain(domain_info[0], is_create_fig=False, color='w')

fs('./Jri-2d')
