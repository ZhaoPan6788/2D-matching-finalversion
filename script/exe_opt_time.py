#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File name: exe_opt_time.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: Submit a task and collect statistics on the execution time of the task.
"""

help_string = """Submit a task and collect statistics on the execution time of the task

usage: 
    python {} [options] [value]

options:
    -s/src      Path to work directory. Default is ".".
    -r/rule     Matching rules for the specified task. Default is ".".
    -n/name     The name of the executable file. Default is "pic2d-mpi".
    -y          If set, the script silent running. Default is not set.
    -serial     if set, serial execution task. Default is not set.
    -h/help     Output help information.

"""
help_string = help_string.format(__file__)

silence_flag = False
serial_flag = False

import os
import sys
import time
import copy
import subprocess
import concurrent.futures

from base.parse_command_line_input import parse_input
from base.remove_files import remove_files
from base.filter_files_with_regex import filter_files_with_regex
from base.create_folder import create_folder
from base.batch_exe import batch_exe
from base.import_json import import_json


if __name__ == '__main__':
    # Paramter
    work_path  = "."
    rule_match = "."
    exe_name = "pic2d-mpi"

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

            if "n" in input_dict:
                exe_name = input_dict["n"]

            if "name" in input_dict:
                exe_name = input_dict["name"]

            if "y" in input_dict:
                silence_flag = input_dict["y"]

            if "serial" in input_dict:
                serial_flag = input_dict["serial"]

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

    # Run
    work_path = os.path.abspath(work_path)
    matched_dirs = filter_files_with_regex(rule_match, work_path)
    matched_dirs.sort(reverse=False)

    executables = []
    if matched_dirs:
        for idir in matched_dirs:
            if os.path.exists(os.path.join(work_path, idir, exe_name)):
                d = import_json(os.path.join(work_path, idir, 'config.json'))
                num_cores = d['grid']['deps_in_z'] * d['grid']['deps_in_r']
                exe_args = ['mpirun', '-np', str(num_cores), os.path.join(work_path, idir, exe_name)]
                executables.append(exe_args)

    else:
        print("The folders is not matched in path [{0}] with the regex [{1}]".format(work_path, rule_match))
        sys.exit(-1)

    print("The following executable file will be executed: ")
    for iexe in executables:
        print(iexe)
    
    if not silence_flag:
        sure = input('Are you sure (yes/no): ')
        if sure not in ['y', 'Y', 'yes', 'Yes']:
            sys.exit(-1)
    print("")

    results = batch_exe(executables, serial_run=serial_flag)
    
    print("Execution time: ")
    for i in range(len(executables)):
        print('    - {}           {}'.format(executables[i][-1], results[i]))
