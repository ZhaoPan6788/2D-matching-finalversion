#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import os
import re
import math
import h5py

import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from base.read_matrix import MatrixDataReader
from geom.geometry import *
from base.common import *

def get_domain_info(path):
    with open(path, 'r') as f:
        data_list = []
        while True:
            line = f.readline()
            if line:
                data_list.append([int(i) for i in line.split()])
            else:
                break

    domain_info = []
    for i in range(len(data_list)):
        domain = data_list[i]
        dim = domain[0]
        s = 1
        domain_shape = domain[s:s+dim]
        s = s + dim
        image_shape = domain[s:s+dim]
        s = s + dim

        local_shape = []
        for i in range(dim):
            local_shape.append(domain[s:s+image_shape[i]])
            s = s + image_shape[i]

        domain_dict = {}
        domain_dict['dim'] = dim
        domain_dict['shape'] = domain_shape
        domain_dict['image'] = image_shape
        domain_dict['local'] = local_shape
        domain_info.append(domain_dict)
    return domain_info


def draw_domain(domain_info, is_create_fig: bool = True, color='gray', isvert=False) -> None:
    if is_create_fig:
        plt.figure(figsize=(12, 8))

    info = domain_info
    Nz = info['shape'][0]
    Nr = info['shape'][1]
    deps_z = info['image'][0]
    deps_r = info['image'][1]
    local = info['local']

    if not isvert:
        s = 0
        for i in range(len(local[0])-1):
            s = s + local[0][i]-1
            plt.hlines(y=s, xmin=0, xmax=Nr-1, color=color, linewidth=0.5)

        s = 0
        for i in range(len(local[1])-1):
            s = s + local[1][i]-1
            plt.vlines(x=s, ymin=0, ymax=Nz-1, color=color, linewidth=0.5)

        if is_create_fig:
            plt.xlim(0, Nr - 1)
            plt.ylim(0, Nz - 1)
            plt.xlabel('R-axis')
            plt.ylabel('Z-axis')
            plt.show()
    else:
        s = 0
        for i in range(len(local[0])-1):
            s = s + local[0][i]-1
            plt.vlines(x=s, ymin=0, ymax=Nr-1, color=color, linewidth=0.5)

        s = 0
        for i in range(len(local[1])-1):
            s = s + local[1][i]-1
            plt.hlines(y=s, xmin=0, xmax=Nz-1, color=color, linewidth=0.5)

        if is_create_fig:
            plt.xlim(0, Nz - 1)
            plt.ylim(0, Nr - 1)
            plt.xlabel('Z-axis')
            plt.ylabel('R-axis')
            plt.show()


def getMin2(data):
    minValue = data.min()
    min2 = minValue
    a = data.flatten()
    for i in range(a.size):
        if a[i] > minValue:
            if abs(min2 - minValue) < 1e-15 or min2 > a[i]:
                min2 = a[i]
    return np.array([minValue, min2])


def getMM(data):
    '''
    判断 data 是否可以绘制成 半对数图   getMinMax(data)
    '''
    minValue = data.min()
    maxValue = data.max()

    if minValue >= 0 and minValue < maxValue:
        temp = data.flatten()
        temp = getMin2(temp)
        if temp[0] == 0:
            tempMin = temp[1]
        else:
            tempMin = temp[0]
        tempMax = maxValue

        if (tempMax / tempMin > 10):
            return True, tempMin, tempMax

    return False, minValue, maxValue


def getMinMax(diagSet, name):
    """
    计算最大值和最小值
    :param diagSet: 诊断数据列表
    :param name: 参量名称
    :return: [min, max]
    """

    diagSetMinValues = []
    diagSetMaxValues = []
    for id in diagSet:
        flag, mint, maxt = getMM(id.getData(name))
        diagSetMinValues.append(np.min(mint))
        diagSetMaxValues.append(np.max(maxt))

    minValue = np.min(diagSetMinValues)
    maxValue = np.max(diagSetMaxValues)

    return minValue, maxValue


fsave_is_save_global = True
fsave_dpi_global = 300
fsave_pad_inches_global = 0.2


def fs(name='fig.svg', is_save=None, dpi=None, pad_inches=None):
    global fsave_is_save_global
    global fsave_dpi_global
    global fsave_pad_inches_global

    _is_save = fsave_is_save_global
    _dpi = fsave_dpi_global
    _pad_inches = fsave_pad_inches_global
    if is_save != None:
        _is_save = is_save

    if dpi != None:
        _dpi = dpi

    if pad_inches != None:
        _pad_inches = pad_inches

    if _is_save:
        plt.savefig(name, dpi=_dpi, pad_inches=_pad_inches)

    else:
        plt.show()


def fsave_set_save(is_save):
    global fsave_is_save_global
    fsave_is_save_global = is_save
    return None


def fsave_set_dpi(dpi):
    global fsave_dpi_global
    fsave_dpi_global = dpi
    return None


def fsave_set_pad_inches(pad_inches):
    global fsave_pad_inches_global
    fsave_pad_inches_global = pad_inches
    return None


def curvestyle(fgsize=(8, 6)):
    plt.figure(figsize=fgsize)
    bwith = 2
    linewidth = 2
    fontsize = 16
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    font_dict = {'size': 24}
    plt.tick_params(direction='out', which='major', length=6,
                    width=linewidth, bottom=True, top=False, left=True, right=False)

    plt.tick_params(direction='out', which='minor', length=4,
                    width=linewidth*0.6, bottom=True, top=False, left=True, right=False)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return font_dict, fontsize


def find_min_positive(data):
    positive_numbers = data[data > 0]

    if len(positive_numbers) == 0:
        return None

    return np.min(positive_numbers)


def contour(xx, yy, zz, lev=120, log=False, cmap='jet', bar=True, z_min=None, z_max=None, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    bwith = 3
    linewidth = 3
    fontsize = 20
    font_dict = {'size': 20}

    plt.tick_params(direction='out', which='major', length=6,
                    width=1.5, bottom=True, top=False, left=True, right=False)

    plt.tick_params(direction='out', which='minor', length=4,
                    width=1.5*0.6, bottom=True, top=False, left=True, right=False)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    zz_min = np.min(zz)
    zz_max = np.max(zz)
    if z_min != None:
        zz_min = z_min
    
    if z_max != None:
        zz_max = z_max

    if not log:
        norm = cm.colors.Normalize(vmin=zz_min, vmax=zz_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        lims = np.linspace(zz_min, zz_max, lev)
        CS = plt.contourf(xx,
                         yy,
                         zz, levels=lims, cmap=cmap, norm=norm)
    else:
        if zz_min <= 0:
            zz_min = find_min_positive(zz)

        norm = cm.colors.LogNorm(vmin=zz_min, vmax=zz_max)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        lims = np.logspace(np.floor(np.log10(zz_min)), np.ceil(np.log10(zz_max)), lev)
        CS = plt.contourf(xx,
                         yy,
                         zz,
                         levels=lims,
                         cmap=cmap, norm=norm)

    ax = plt.gca()
    ax.spines['bottom'].set_color('0.0')
    ax.spines['top'].set_color('0.0')
    ax.spines['right'].set_color('0.0')
    ax.spines['left'].set_color('0.0')
    ax.spines['bottom'].set_linewidth(1.5)     # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)       # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)      # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)        # 设置上部坐标轴的粗细
    plt.grid(False)

    if (bar):
        cb = plt.colorbar(sm, fraction=0.15, pad=0.1, boundaries=lims)
        cb.ax.tick_params(labelsize=fontsize)     # 设置色标刻度字体大小
        cb.ax.tick_params(length=6, width=1.5)
        cb.outline.set_linewidth(1.5)

    return font_dict, fontsize, sm, lims


colors = ['#f85a40', '#7552cc', '#00c16e', '#037ef3',
          '#ffc845', '#f48924', '#ffc845', '#52565e', '#caccd1']


def substyle():
    bwith = 3
    linewidth = 3
    fontsize = 20
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    font_dict = {'size': 20}

    return font_dict, fontsize


def sub2(ax1, ax2, font, size):
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, prop=font, frameon=False)

    # 设置刻度字体大小
    ax1.tick_params(direction='in', which='major', length=6, labelsize=size,
                    width=3, bottom=True, top=True, left=True, right=False)

    ax1.tick_params(direction='in', which='minor', length=4, labelsize=size,
                    width=3*0.6, bottom=True, top=True, left=True, right=False)

    ax2.tick_params(direction='in', which='major', length=6, labelsize=size,
                    width=3, bottom=False, top=False, left=False, right=True)

    ax2.tick_params(direction='in', which='minor', length=4, labelsize=size,
                    width=3*0.6, bottom=False, top=False, left=False, right=True)


def drawcontourf2D(zz, rr, data, dmin=None, dmax=None, lev=120, fsave=None, tlt='', xlb='R (cm)', ylb='Z (cm)', unit='Density ($m^{-3}$)'):
    font_dict = {'size': 20}

    data_min = np.min(data)
    if None != dmin:
        data_min = dmin

    data_max = np.max(data)
    if None != dmax:
        data_max = dmax

    plt.figure(figsize=(8, 6))
    lims = np.linspace(data_min, data_max, lev)
    CS = plt.contourf(zz, rr, data, lims, cmap='plasma')

    cb = plt.colorbar(CS)
    ax = plt.gca()
    ax.spines['bottom'].set_color('0.0')
    ax.spines['top'].set_color('0.0')
    ax.spines['right'].set_color('0.0')
    ax.spines['left'].set_color('0.0')
    ax.spines['bottom'].set_linewidth(3/5)     # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3/5)       # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3/5)      # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3/5)        # 设置上部坐标轴的粗细

    plt.grid(False)
    plt.xlabel(xlb, fontdict=font_dict)
    plt.ylabel(ylb, fontdict=font_dict)
    plt.title(tlt)

    cb.ax.tick_params(labelsize=10)     # 设置色标刻度字体大小
    cb.outline.set_visible(False)
    cb.set_label(unit, fontdict=font_dict)  # 设置colorbar的标签字体及其大小

    if None == fsave:
        plt.show()
    else:
        plt.savefig(fsave, dpi=100, pad_inches=0.2)


def calculate_electric_field(potential, dz, dr):
    shape = potential.shape
    electric_field_z = np.zeros(shape)
    electric_field_r = np.zeros(shape)

    # 计算R方向的电场
    electric_field_r[1:-1, :] = (potential[2:, :] -
                                 potential[:-2, :]) / (2 * dr)
    electric_field_r[0, :] = (4 * potential[1, :] -
                              3 * potential[0, :] - potential[2, :]) / (2 * dr)
    electric_field_r[-1, :] = (4 * potential[-2, :] -
                               3 * potential[-1, :] - potential[-3, :]) / (2 * dr)

    # 计算Z方向的电场
    electric_field_z[:, 1:-1] = (potential[:, 2:] -
                                 potential[:, :-2]) / (2 * dz)
    electric_field_z[:, 0] = (4 * potential[:, 1] -
                              3 * potential[:, 0] - potential[:, 2]) / (2 * dz)
    electric_field_z[:, -1] = (4 * potential[:, -2] -
                               3 * potential[:, -1] - potential[:, -3]) / (2 * dz)

    return -electric_field_z, -electric_field_r


def get_mean_field(h5files, varname, istart=0, iend=-1):
    hdcopy = h5files[istart:iend]
    data = np.array(hdcopy[0][varname])
    for i in range(1, len(hdcopy)):
        data = data + np.array(hdcopy[i][varname])

    return data / len(hdcopy)


class H5readPF:
    def __init__(self, file_path, filestr, dz=1, dr=1, name='geom'):
        self.sim_path = os.path.abspath(file_path)
        self.pf_str = filestr
        self.lb = os.path.split(self.sim_path)[-1]
        self.dz = dz
        self.dr = dr

        self.domain_info = get_domain_info(os.path.join(self.sim_path, 'check/domain.dat'))
        self.g = Geometry(1, 1)
        self.g.load(os.path.join(self.sim_path, 'input'), name=name)
        self.Nz = self.g.Nz
        self.Nr = self.g.Nr

        self.zpos = np.arange(self.Nz)
        self.rpos = np.arange(self.Nr)
        [self.rr, self.zz] = np.meshgrid(self.rpos, self.zpos)

        self.sim_path = os.path.join(self.sim_path, 'diag')
        self.subDirs = os.listdir(self.sim_path)
        self.subDirs.sort(reverse=False)

        if len(self.subDirs) > 0:
            tmpDir = list()

            for i in range(len(self.subDirs)):
                if len(re.findall(self.pf_str+'_\d*[.]h5', self.subDirs[i])) > 0:
                    tmpDir.append(self.subDirs[i])

            if len(tmpDir) > 0:
                self.subDirs = tmpDir
            else:
                print('The number of directories that meet the rule is 0.')
                exit(-1)

        print('\nH5 file number: ', len(self.subDirs))
        print('')

        self.h5files = []

    def read_h5(self, index_start=None, index_stop=None):
        if index_start == None:
            index_start = 0
        
        if index_stop == None:
            index_stop = len(self.subDirs)

        self.close_all()
        self.h5files = []
        ss = self.subDirs[index_start:index_stop]
        for i in range(len(ss)):
            self.h5files.append(h5py.File(os.path.join(self.sim_path, ss[i]), 'r'))

    def read_mean(self, index_start=None, index_stop=None):
        if index_start == None:
            index_start = 0
        
        if index_stop == None:
            index_stop = len(self.subDirs)
        
        ss = self.subDirs[index_start:index_stop]

        self.time = np.array([i for i in range(len(ss))])
        self.ne_mean = np.zeros((len(ss), 1))
        self.ni_mean = np.zeros((len(ss), 1))
        self.Te_mean = np.zeros((len(ss), 1))
        self.Ti_mean = np.zeros((len(ss), 1))
        self.phi = np.zeros((len(ss), 1))

        ro = 100
        for j in range(math.ceil(len(ss)/ro)):
            start = j * ro
            stop = (j + 1) * ro
            if stop > len(ss):
                stop = len(ss)

            h5files = []
            for i in range(start, stop):
                h5files.append(h5py.File(os.path.join(self.sim_path, ss[i]), 'r'))

            for i in range(len(h5files)):
                self.ne_mean[i+start] = np.mean(h5files[i]['RhoOne-Electron'])
                self.ni_mean[i+start] = np.mean(h5files[i]['RhoOne-Ar+'])
                self.Te_mean[i+start] = np.mean(h5files[i]['EnergyOne-Electron'])
                self.Ti_mean[i+start] = np.mean(h5files[i]['EnergyOne-Ar+'])
                self.phi[i+start] = h5files[i]['Phi'][0, -1]

            for i in range(len(h5files)):
                h5files[i].close()

    def close_all(self):
        for i in range(len(self.h5files)):
            self.h5files[i].close()
            
            
linestyle_tuple = [('solid', 'solid'),
                   ('dotted', 'dotted'),
                   ('dashed', 'dashed'),
                   ('dashdot', 'dashdot'),
                   ('loosely dotted', (0, (1, 10))),
                   ('dotted', (0, (1, 1))),
                   ('densely dotted', (0, (1, 2))),
                   ('loosely dashed', (0, (5, 10))),
                   ('dashed', (0, (5, 5))),
                   ('densely dashed', (0, (5, 1))),
                   ('loosely dashdotted', (0, (3, 10, 1, 10))),
                   ('dashdotted', (0, (3, 5, 1, 5))),
                   ('densely dashdotted', (0, (3, 1, 1, 1))),
                   ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
                   ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                   ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def run_command(cmd, work_dir=None):
    """
    执行系统 shell 命令的通用封装。
    
    :param cmd: 要执行的命令字符串 (例如: 'mpirun -np 4 ./bin/ZPrun2D')
    :param work_dir: (可选) 执行命令的工作目录。如果不指定，则在当前目录执行。
    :return: 命令执行的返回码 (0 表示成功)
    """
    import os
    
    # 记录当前目录，以便执行后切回
    cwd = os.getcwd()
    
    # 如果指定了工作目录，先切换过去
    if work_dir:
        if not os.path.exists(work_dir):
            print(f"[Error] Work directory not found: {work_dir}")
            return -1
        os.chdir(work_dir)
        
    print(f"Executing: {cmd}")
    
    # 执行命令
    # 注意：这里使用了 os.system，为了更高级的控制也可以换成 subprocess.run
    ret = os.system(cmd)
    
    if ret != 0:
        print(f"[Warning] Command '{cmd}' returned non-zero exit code: {ret}")
        
    # 切回原目录
    if work_dir:
        os.chdir(cwd)
        
    return ret