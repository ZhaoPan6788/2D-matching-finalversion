#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: generate_pbs.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-08-28
Description: A script used to generate pbs related files.
"""

import os
import sys

def generate_pbs(path, pbs_name, task_name='picmcc', queue='slim', nodes=1, ppn=16, launch_file='launch.py'):
    """

    """

    pbs_str = ''
    pbs_str = pbs_str + '#!/bin/sh\n'
    pbs_str = pbs_str + '#PBS -N {}\n'.format(task_name)
    pbs_str = pbs_str + '#PBS -q {}\n'.format(queue)
    pbs_str = pbs_str + '#PBS -l nodes={}:ppn={},walltime=05:00:00:00\n'.format(nodes, int(ppn))
    pbs_str = pbs_str + '#PBS -V\n'
    pbs_str = pbs_str + '#PBS -S /bin/bash\n'
    pbs_str = pbs_str + '#PBS -j oe\n'
    pbs_str = pbs_str + '#PBS -o ./out.log\n'
    pbs_str = pbs_str + '\n'
    pbs_str = pbs_str + 'EXEPATH=$PBS_O_WORKDIR\n'
    pbs_str = pbs_str + 'EXEC="$EXEPATH/{}"\n'.format(launch_file)
    pbs_str = pbs_str + 'cd $EXEPATH\n'
    pbs_str = pbs_str + '\n'
    pbs_str = pbs_str + 'NP=`cat $PBS_NODEFILE | wc -l`\n'
    pbs_str = pbs_str + 'NN=`cat $PBS_NODEFILE | sort | uniq | tee ./nodes.txt | wc -l`\n'
    pbs_str = pbs_str + 'cat $PBS_NODEFILE > ./nodefile.txt\n'
    pbs_str = pbs_str + '\n'
    pbs_str = pbs_str + 'python $EXEC\n'

    with open(os.path.join(path, pbs_name), 'w') as f:
        f.write(pbs_str)


def generate_slurm_simple(path, slurm_name, nodes=1, launch_file='launch.py'):
    """

    """

    slurm_str = ''
    slurm_str = slurm_str + '#!/bin/bash\n'
    slurm_str = slurm_str + '#SBATCH -N {}\n'.format(nodes)
    slurm_str = slurm_str + 'module load ~/mpimodule\n'
    slurm_str = slurm_str + 'module load python/3.7.6\n'
    slurm_str = slurm_str + 'cd {}\n'.format(path)
    slurm_str = slurm_str + 'python {} &\n'.format(launch_file)
    slurm_str = slurm_str + 'wait\n'.format(launch_file)

    with open(os.path.join(path, slurm_name), 'w') as f:
        f.write(slurm_str)


def generate_slurm(path, slurm_name, p, nodes=1, cpus=64, ntasks=1, launch_file='launch.py'):
    """

    """
    task_name = os.path.split(path)[-1]

    slurm_str = ''
    slurm_str = slurm_str + '#!/bin/sh\n'
    slurm_str = slurm_str + '#SBATCH --job-name={}\n'.format(task_name)
    slurm_str = slurm_str + '#SBATCH --nodes={}\n'.format(nodes)
    slurm_str = slurm_str + '#SBATCH --ntasks-per-node={}\n'.format(ntasks)
    slurm_str = slurm_str + '#SBATCH --cpus-per-task={}\n'.format(cpus)
    slurm_str = slurm_str + '#SBATCH --partition={}\n'.format(p)
    slurm_str = slurm_str + 'module purge\n'
    slurm_str = slurm_str + 'module load mpi/intelmpi/2021.3.0\n'
    slurm_str = slurm_str + 'cd {}\n'.format(path)
    slurm_str = slurm_str + 'python {} &\n'.format(launch_file)
    slurm_str = slurm_str + 'wait\n'.format(launch_file)

    with open(os.path.join(path, slurm_name), 'w') as f:
        f.write(slurm_str)


def generate_launch(base_path, launch_path, launch_name, task_paths):
    """

    """

    launch_str = ''
    launch_str = launch_str + 'import sys\n'
    launch_str = launch_str + 'sys.path.append("{}")\n'.format(base_path)
    launch_str = launch_str + 'from base.batch_exe import batch_exe\n'
    launch_str = launch_str + '\n'
    launch_str = launch_str + 'results = batch_exe({})\n'.format(task_paths)
    launch_str = launch_str + 'print("Execution time: ")\n'
    launch_str = launch_str + 'for i in range(len({})):\n'.format(task_paths)
    launch_str = launch_str + '    print("    - {{}}           {{}}".format({0}[i][-1], results[i]))\n'.format(task_paths)

    with open(os.path.join(launch_path, launch_name), 'w') as f:
        f.write(launch_str)
