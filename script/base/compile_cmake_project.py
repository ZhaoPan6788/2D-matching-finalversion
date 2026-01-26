#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: compile_cmake_project.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-04
Description: A python script to compile CMake project in Linux.
"""

import os
import shutil
import subprocess

def compile_cmake_project(src_path, fortran_compiler=None, c_compiler=None, cpp_compiler=None, cpu_cores=8):
    """
    Arguments:
        src_path (str): The source path of the project.
        fortran_compiler (str, optional): The Fortran compiler to use.
        c_compiler (str, optional): The C compiler to use.
        cpp_compiler (str, optional): The C++ compiler to use.
    
    Returns:
        bool: True if the project is compiled successfully, False otherwise.
        
    Description:
        Compile a CMake project in Linux. The source path and compilers are specified as arguments.
    """
    if not os.path.exists(src_path):
        print("Source path does not exist.")
        return False
    
    raw_path = os.getcwd()
    os.chdir(src_path)

    build_dir = os.path.join(src_path, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    cmake_command = ["cmake", "-S", src_path, "-B", build_dir]
    if fortran_compiler:
        cmake_command.extend(["-DCMAKE_Fortran_COMPILER={}".format(fortran_compiler)])
    if c_compiler:
        cmake_command.extend(["-DCMAKE_C_COMPILER={}".format(c_compiler)])
    if cpp_compiler:
        cmake_command.extend(["-DCMAKE_CXX_COMPILER={}".format(cpp_compiler)])
    
    subprocess.run(cmake_command)
    os.environ["MAKEFLAGS"] = "-j {}".format(cpu_cores)
    os.chdir(build_dir)
    subprocess.run("make")
    # subprocess.run(["cmake", "--build", build_dir])

    os.chdir(raw_path)
    print("Project is compiled successfully at {}".format(build_dir))
    return True

if __name__ == '__main__':
    compile_cmake_project('/home/chenzili/workspace/pic/ccp/iPM2D', 'ifort', 'icc', 'icpc', 8)
