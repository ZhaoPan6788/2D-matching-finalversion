#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File: read_matrix.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-03-15
Description: Read matrix data in text format.
"""

import os
import time
import numpy as np
import pandas as pd


# Custom function to convert strings to numbers and replace non-convertible strings with 0
def str_to_num(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


class MatrixDataReader:
    def __init__(self, path_list):
        if not isinstance(path_list, list):
            self.path_list = [path_list]
        else:
            self.path_list = path_list

        self.is_print = True
        self.data = []

    def read_matrices(self):
        self.data = []
        for path in self.path_list:
            if not os.path.exists(path):
                print('Path "{}" does not exist. Skipping.'.format(path))
                continue

            start_time = time.time()

            # vecDR = dt.fread(path, header=False)
            data_pd = pd.read_csv(path, sep='\s+', header=None)

            # Convert datatable to pandas DataFrame
            # data_pd = vecDR.to_pandas()

            # Iterate over columns
            for col in data_pd.columns:
                # If the column type is 'object' (string), apply the custom function to convert strings to numbers
                if data_pd[col].dtype == 'object':
                    data_pd[col] = data_pd[col].apply(str_to_num)
                else:
                    # Replace non-numeric values (NaN) with 0
                    data_pd[col] = data_pd[col].fillna(0)

            vecDR = data_pd.values
            matrix = [vecDR[:, i] for i in range(vecDR.shape[1])]
            vecDR = []

            num_rows = len(matrix)
            num_cols = len(matrix[0])
            file_size = os.path.getsize(path)

            elapsed_time = time.time() - start_time

            if self.is_print:
                print('File: {}\n'
                    'Rows: {}\n'
                    'Cols: {}\n'
                    'Size: {} bytes\n'
                    'Time: {:.2f} seconds\n'.format(path, num_rows, num_cols, file_size, elapsed_time))

            self.data.append(matrix)

# =============================================================================
# Append this class to iPM2D-grid30/script/base/read_matrix.py
# =============================================================================

import copy
import numpy as np
from base.impedance_analysis import get_equivalent_impedance

class PlasmaParameters:
    def __init__(self, MatrixData, d):
        """
        MatrixData: MatrixDataReader 实例，包含读取到的 EC.txt 数据
        d: config.json 的配置字典
        """
        self.d = copy.deepcopy(d)
        self.valid = False

        # 1. 检查数据有效性
        # MatrixData.data 是一个列表的列表，data[0] 对应读取的第一个文件(EC.txt)
        if not MatrixData.data or not MatrixData.data[0]:
            print("Error: No data loaded from EC.txt")
            return
        
        # 获取第一个文件的数据矩阵
        ec_data = MatrixData.data[0]
        rows = len(ec_data[0]) # 数据行数（时间步数）
        cols = len(ec_data)    # 数据列数
        
        # 2. 解析物理量 (基于 Circuits.F90 的输出格式)
        # Col 0: Time
        # Col 1: Electrode Voltage (MT%metls%voltage)
        # Col 11: Electrode Current (imn_electrode_current) -> 对应 Fortran write 的第 12 个变量
        
        if cols < 12:
            print(f"Error: EC.txt has only {cols} columns, expected at least 12.")
            return

        self.time = ec_data[0]
        self.V_ele = ec_data[1]  # 负载电压
        self.I_ele = ec_data[11] # 负载电流 (注意：Python索引从0开始，所以第12列是index 11)

        # 计算时间步长 dt
        if len(self.time) > 1:
            self.dt = self.time[1] - self.time[0]
        else:
            self.dt = 1e-12

        # 3. 读取配置参数
        # 尝试适配 2D 的 config.json 结构
        try:
            # 频率
            if 'freq' in self.d['circuit']:
                self.freq = self.d['circuit']['freq']
            elif 'imn' in self.d['circuit'] and 'freq' in self.d['circuit']['imn']: # 备用路径
                self.freq = self.d['circuit']['imn']['freq']
            else:
                self.freq = 13.56e6 # 默认值

            # 射频源内阻
            self.Rs = self.d['circuit'].get('Rs', 50.0)

            # 匹配网络固定元件 (用于 impedance_analysis 计算)
            # 根据 IMN.F90 拓扑: Lm, Rm, Cm1, Cm2
            imn_cfg = self.d['circuit'].get('imn', {})
            self.Lm = imn_cfg.get('Lm', 0.0)
            self.Rm = imn_cfg.get('Rm', 0.0)
            
            # Cm1, Cm2 主要在优化器中作为变量，但这里也可以存一下初始值
            self.Cm1 = imn_cfg.get('Cm1', 1000e-12)
            self.Cm2 = imn_cfg.get('Cm2', 100e-12)

        except Exception as e:
            print(f"Warning: Error reading config parameters: {e}")
            self.freq = 13.56e6
            self.Rs = 50.0
            self.Lm = 0.0
            self.Rm = 0.0

        # 4. 计算等效负载阻抗 Z_load
        # 截取最后 N 个周期的数据进行计算，以保证处于稳态
        n_cycles = 20
        period = 1.0 / self.freq if self.freq > 0 else 1.0
        time_window = n_cycles * period
        
        # 确定截取范围
        if self.time[-1] > time_window:
            t_end = self.time[-1]
            t_start = t_end - time_window
            # 使用布尔掩码截取
            mask = self.time >= t_start
            v_sample = self.V_ele[mask]
            i_sample = self.I_ele[mask]
        else:
            # 如果数据不够长，就用全部数据
            v_sample = self.V_ele
            i_sample = self.I_ele

        # 调用 impedance_analysis 中的函数计算阻抗
        self.Z_load = get_equivalent_impedance(v_sample, i_sample, self.dt, self.freq)
        
        self.valid = True

