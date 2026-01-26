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

