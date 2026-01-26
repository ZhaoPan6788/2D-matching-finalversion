#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File name: batch_exe.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: Functions for batch submitting tasks.
"""

import os
import sys
import time
import copy
import subprocess
import concurrent.futures


def run_executable(executable):
    """
    Arguments:
        executable (str): The name of the executable file to be run

    Returns:
        Real: Returns total execution time
        
    Description:
        This function runs the specified executable file with the given arguments, and measures the execution time.
        It also prints information about the program's start and end.
    """

    start_time = time.time()
    print("Program '{}' \n        started at {}\n".format(executable[-1], time.ctime(start_time)))

    directory = os.path.dirname(executable[-1])
    os.chdir(directory)
    
    with open(os.path.join(directory, 'run.log'), 'w') as f:
        p = subprocess.Popen(' '.join(executable), shell = True, stdout=f, universal_newlines=True)
        p.communicate()
        p.wait()

    end_time = time.time()
    print("Program '{}' \n        finished at {}".format(executable[-1], time.ctime(end_time)))
    print("Total execution time: {:.2f} seconds".format(end_time - start_time))
    
    return end_time - start_time


def batch_exe(task_abs_paths, serial_run=False):
    """
    Arguments:
        task_abs_paths (list): A list of absolute paths of the executable files to be executed
        serial_run (bool, optional): Whether to run the executable files in serial or parallel mode. Defaults to False.
    
    Returns:
        list: A list of the results of executing each executable file.
        
    Description:
        Execute a list of executable files using either parallel or serial mode. The function uses a ProcessPoolExecutor to execute the 
        files in parallel mode, and runs the files in serial mode otherwise. The results of executing each file are stored in a list and 
        returned.
    """

    results = []
    if not serial_run:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(run_executable, executable) for executable in task_abs_paths]
            results = [res.result() for res in results]
    else:
        results = [run_executable(executable) for executable in task_abs_paths]

    return results
