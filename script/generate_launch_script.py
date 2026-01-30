#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File name: generate_launch_script.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: Generate a script to launch the executable files for each task.
"""

help_string = """Generate a script to launch the executable files for each task

usage: 
    python {} [options] [value]

options:
    -s/src      Path to work directory. Default is ".".
    -r/rule     Matching rules for the specified task. Default is ".".
    -n/name     The name of the launch file. Default is "launch.py".
    -j/json     The name of the configuration file. Default is "config.json".
    -e/exe      The name of the executable file. Default is "pic2d-mpi".
    -pbs        If set, the pbs script is generated. Default is not set.
    -slurm      If set, the slurm script is generated. Default is not set.
    -par        Partition name for slurm. Default is "test".
    -m/merge    If set, automatically merge tasks based on number of cores. Default is not set.
    -c/current  If set, generate a launch script for the current directory. Default is not set.
    -max_core   The maximum number of cores per node. Default is 64.
    -match      If set, generate a launch script for impedance matching. Default is not set.
    -y          If set, the script silent running. Default is not set.
    -h/help     Output help information.

"""
help_string = help_string.format(__file__)

silence_flag = False

import os
import sys
import time
import math
import copy
import subprocess
import concurrent.futures

from base.parse_command_line_input import parse_input
from base.remove_files import remove_files
from base.filter_files_with_regex import filter_files_with_regex
from base.create_folder import create_folder
from base.batch_exe import batch_exe
from base.generate_launch import generate_launch, generate_pbs, generate_slurm, generate_match
from base.import_json import import_json
from base.copy_file_or_folder import copy_file_or_folder
from base.integer_grouping import integer_grouping

def get_nodes_ppn(core, max_core):
    """
    
    """
    nodes_min = math.ceil(core / max_core)
    for i in range(nodes_min, math.floor(math.sqrt(core))+1):
        if core % i == 0:
            return i, core // i

    return -1, -1


if __name__ == '__main__':
    # Paramter
    work_path  = "."
    rule_match = "."
    launch_name = "launch.py"
    config_name = "config.json"
    exe_name = "pic2d-mpi"
    is_pbs = False
    is_slurm = False
    par_name = "test"
    
    is_match = False

    merge_tasks = False
    is_current = False

    max_core = 64

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
                launch_name = input_dict["n"]

            if "name" in input_dict:
                launch_name = input_dict["name"]

            if "j" in input_dict:
                config_name = input_dict["j"]

            if "json" in input_dict:
                config_name = input_dict["json"]

            if "e" in input_dict:
                exe_name = input_dict["e"]

            if "exe" in input_dict:
                exe_name = input_dict["exe"]

            if "pbs" in input_dict:
                is_pbs = True

            if "slurm" in input_dict:
                is_slurm = True

            if "par" in input_dict:
                par_name = input_dict["par"]

            if "m" in input_dict:
                merge_tasks = input_dict["m"]

            if "merge" in input_dict:
                merge_tasks = input_dict["merge"]

            if "c" in input_dict:
                is_current = input_dict["c"]

            if "current" in input_dict:
                is_current = input_dict["current"]

            if "max_core" in input_dict:
                max_core = input_dict["max_core"]

            if "match" in input_dict:
                is_match = True

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

    # dirs
    if is_current:
        matched_dirs = [os.getcwd()]
    else:
        work_path = os.path.abspath(work_path)
        matched_dirs = filter_files_with_regex(rule_match, work_path)
        matched_dirs.sort(reverse=False)
        matched_dirs = [os.path.join(work_path, idir) for idir in matched_dirs]

    print("The launch script will be generated in the following directory: ")
    for idir in matched_dirs:
        print('    - '+idir)
    
    if not silence_flag:
        sure = input('Are you sure (yes/no): ')
        if sure not in ['y', 'Y', 'yes', 'Yes']:
            sys.exit(-1)
    print("")

    # generate
    if matched_dirs:
        if not merge_tasks:
            for idir in matched_dirs:
                json_data = import_json(os.path.join(idir, config_name))
                core_num = json_data['grid']['deps_in_z'] * json_data['grid']['deps_in_r']
                # if is_slurm:
                #     exe_args = ['srun', '-n '+str(core_num), '--exclusive -c 1', os.path.join(idir, exe_name)]
                # else:
                exe_args = ['mpirun', '-np', str(core_num), os.path.join(idir, exe_name)]

                if is_match:
                    generate_match(os.path.split(os.path.realpath(__file__))[0], 
                                    idir, launch_name, [exe_args])
                else:
                    generate_launch(os.path.split(os.path.realpath(__file__))[0], 
                                    idir, launch_name, [exe_args])

                nodes, ppn = get_nodes_ppn(core_num, max_core)

                if is_pbs:
                    queue = 'slim'
                    if core_num > max_core:
                        queue = 'regular'
                    
                    if nodes <= 0:
                        print("The number of cores needs to be evenly distributed across each node.")
                        sys.exit(-1)
                    generate_pbs(idir, 'launch.pbs', queue=queue, 
                                nodes=nodes, 
                                ppn=ppn, 
                                launch_file=launch_name)
                
                if is_slurm:
                    generate_slurm(idir, 'launch.slurm', par_name,
                                   nodes, ppn, 1, launch_name)

        else:
            # get core number of each task
            cores = []
            for idir in matched_dirs:
                json_data = import_json(os.path.join(idir, config_name))
                core_num = json_data['grid']['deps_in_z'] * json_data['grid']['deps_in_r']
                cores.append(core_num)
            
            # generate tasks
            _, index_groups, _ = integer_grouping(cores, max_core)

            print("The tasks are grouped as follows: ")
            group_cores_sum = []
            for i in range(len(index_groups)):
                cores_sum = 0
                for j in range(len(index_groups[i])):
                    index = index_groups[i][j]
                    cores_sum = cores_sum + cores[index]

                    if j == len(index_groups[i])-1:
                        print("{1}    {0:>4}  {2:>4}".format(cores[index], matched_dirs[index], cores_sum))
                    else:
                        print("{1}    {0:>4}".format(cores[index], matched_dirs[index]))
                
                group_cores_sum.append(cores_sum)
                print("")

            if not silence_flag:
                sure = input('Are you sure (yes/no): ')
                if sure not in ['y', 'Y', 'yes', 'Yes']:
                    sys.exit(-1)
            print("")

            for i in range(len(index_groups)):
                task_path_tmp = os.path.join(os.path.dirname(matched_dirs[0]), 'task_{:0>2}'.format(i))
                create_folder(os.path.dirname(matched_dirs[0]), 'task_{:0>2}'.format(i))

                exes = []
                for j in range(len(index_groups[i])):
                    # if is_slurm:
                    #     exe_args = ['srun', '-n '+str(cores[index_groups[i][j]]), '--exclusive -c 1', os.path.join(matched_dirs[index_groups[i][j]], exe_name)]
                    # else:
                    exe_args = ['mpirun', '-np', str(cores[index_groups[i][j]]), os.path.join(matched_dirs[index_groups[i][j]], exe_name)]

                    exes.append(exe_args)

                if is_match:
                    generate_match(os.path.split(os.path.realpath(__file__))[0], 
                                    task_path_tmp, launch_name, exes)
                else:
                    generate_launch(os.path.split(os.path.realpath(__file__))[0], 
                                    task_path_tmp, launch_name, exes)

                nodes, ppn = get_nodes_ppn(group_cores_sum[i], max_core)
                if is_pbs:
                    queue = 'slim'
                    if group_cores_sum[i] > max_core:
                        queue = 'regular'

                    if nodes <= 0:
                        print("The number of cores needs to be evenly distributed across each node.")
                        sys.exit(-1)

                    generate_pbs(task_path_tmp, 'launch.pbs', queue=queue, 
                                 nodes=nodes, 
                                 ppn=ppn, 
                                 launch_file=launch_name)
                if is_slurm:
                    generate_slurm(task_path_tmp, 'launch.slurm', par_name,
                                   nodes, ppn, 1, launch_name)

    else:
        print("The folders is not matched in path [{0}] with the regex [{1}]".format(work_path, rule_match))
        sys.exit(-1)

    print('Generating completed.')
