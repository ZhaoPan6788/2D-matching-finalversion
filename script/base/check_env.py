#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
File name: check_env.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: Check multiple environment variables in Linux.
"""

import os

def check_env_vars(env_var_list):
    """
    Arguments:
        env_var_list (list[str]): list of environment variables to check.

    Returns:
        bool: return True if all environment variables exist, otherwise return False.

    Description:
        Check multiple environment variables.
    """
    all_exist = True
    for env_var in env_var_list:
        if env_var in os.environ:
            print("Environment variable {} exists, value is {}".format(env_var, os.environ[env_var]))
        else:
            print("Environment variable {} does not exist".format(env_var))
            all_exist = False
    return all_exist

if __name__ == "__main__":
    env_var_list = ["HDF5_ENV_PATH", "PETSC_ENV_PATH", "FFTW_ENV_PATH"]
    if check_env_vars(env_var_list):
        print("All environment variables exist")
    else:
        print("Some environment variables do not exist")
