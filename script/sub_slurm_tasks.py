#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File name: sub_slurm_tasks.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: Batch submit slurm tasks.
"""

help_string = """Batch submit pbs tasks

usage: 
    python {} [options] [value]

options:
    -s/src      Path to work directory. Default is ".".
    -r/rule     Matching rules for the specified task. Default is ".".
    -slurm      The name of the slurm script. Default is "launch.slurm".
    -y          If set, the script silent running. Default is not set.
    -h/help     Output help information.

"""
help_string = help_string.format(__file__)

silence_flag = False

import sys
import os
import re
import math
import functools
import time
import copy
import subprocess
import concurrent.futures

from base.parse_command_line_input import parse_input
from base.remove_files import remove_files
from base.filter_files_with_regex import filter_files_with_regex
from base.create_folder import create_folder
from base.copy_file_or_folder import copy_file_or_folder
from base.import_json import import_json


def checkOne(subList):
    if (len(subList) < 0):
        print('No task to submit.')
        return False

    print('Sub task list : ')
    print('')
    for i in range(len(subList)):
        print(subList[i])
    
    print('')
    sure = input('Are you sure to submit the tasks (yes/no): ')
    return sure in ['y', 'Y', 'yes', 'Yes']


if __name__ == '__main__':
    # Parameter
    work_path = '.'
    rule_match = '.'
    slurm_name = 'launch.slurm'

    # Parse input
    input_dict = parse_input(sys.argv)
    if input_dict:
        try:
            if "s" in input_dict:
                work_path = input_dict["s"]

            if "src" in input_dict:
                work_path = input_dict["src"]

            if "r" in input_dict:
                rule_match = input_dict["r"]

            if "rule" in input_dict:
                rule_match = input_dict["rule"]

            if "slurm" in input_dict:
                slurm_name = input_dict["slurm"]

            if "y" in input_dict:
                silence_flag = input_dict["y"]

            if "h" in input_dict:
                print(help_string)
                sys.exit(-1)

            if "help" in input_dict:
                print(help_string)
                sys.exit(-1)

        except KeyError:
            print("The Key not found in the dictionary.")
        
        print("==> Parse input")
        print("Parsed command line input : ")
        for key, val in input_dict.items():
            print("\t{}: {}".format(key, val))

    print("")

    # task dirs
    work_path = os.path.abspath(work_path)
    subDirs = filter_files_with_regex(rule_match, path=work_path)
    subDirs.sort(reverse=False)

    if not subDirs:
        print('The number of subdirectories under the path is 0.')
        sys.exit(-1)

    # generate sub cmd for task
    cmd_list = []
    for i in range(len(subDirs)):
        cmd_str = ['sbatch', 'launch.slurm']
        cmd_list.append(' '.join(cmd_str))

    if checkOne(cmd_list):
        print('')
        for i in range(len(cmd_list)):
            os.chdir(os.path.join(work_path, subDirs[i]))
            print('Task: ', i)
            print("Enter:    ", os.getcwd())

            os.system(cmd_list[i])
            print("Sub task: ", cmd_list[i])
            print("")
