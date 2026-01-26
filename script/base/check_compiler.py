#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: check_compiler.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: Check if Intel oneAPI compilers exist.
"""

import sys
import os

def check_compiler(compilers):
    """
    Arguments:
        compilers (list[str]): A list of compilers to check.

    Returns:
        bool: True if all compilers exist, False otherwise.
        
    Description:
        Check if all the compilers exist.
    """
    result = True
    for compiler in compilers:
        exists = False
        try:
            compiler_path = os.popen("which {}".format(compiler)).read().strip()
            if os.path.exists(compiler_path):
                exists = True
        except Exception as e:
            exists = False
        result = result and exists
        print("{}: {}".format(compiler, "Exists" if exists else "Not Exists"))
    return result

if __name__ == "__main__":
    compilers = ["icc", "ifort", "icpc"]
    if check_compiler(compilers):
        print("All compilers exist.")
    else:
        print("Some compilers do not exist.")
