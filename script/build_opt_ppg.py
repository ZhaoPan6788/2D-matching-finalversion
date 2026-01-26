#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: build_opt_cores.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: Build the Cmake project to optimize the core.
"""

import os
import sys
import time
import copy

from base.batch_build import build_cmake_project


def generate_tasks_opt_ppg(json_data):
    """
    Arguments:
        json_data (dict): The dictionary object from the JSON file.
    
    Returns:
        dict: The dictionary object imported from the JSON file.
        
    Description:
        Import a JSON file and return the dictionary object.    
    """

    # parameter from Implicit and electrostatic particle-in-cell/Monte Carlo model 
    # in two-dimensional and axisymmetric geometry: I. Analysis of numerical techniques
    # https://iopscience.iop.org/article/10.1088/0963-0252/19/4/045023/meta

    ppg = [100, 200, 400]
    task_name = ['ppg_{}'.format(ippg) for ippg in ppg]

    json_list = []
    try:
        for i in range(len(task_name)):
            d = copy.deepcopy(json_data)

            # parameter
            d["plasma"]["ppg"] = ppg[i]
            json_list.append(d)

    except KeyError:
        print("The Key not found in the dictionary.")

    return json_list, task_name


if __name__ == '__main__':
    build_cmake_project(compilers=["mpiifort", "mpiicc", "mpiicpc"], 
                        env_var_list=["HDF5_ENV_PATH", "PETSC_ENV_PATH", "SUNDIALS_ENV_PATH"], 
                        generate_tasks_function=generate_tasks_opt_ppg)
