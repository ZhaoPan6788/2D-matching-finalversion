#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: create_folder.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: A script to create a folder in the specified path.
"""

import os
import sys

def create_folder(path='.', folder_name='.iPM'):
    """
    Arguments:
        path (str): The path where the folder will be created. (default is '.')
        folder_name (str): The name of the folder to be created. (default is '.iPM')
    
    Returns:
        bool: Return True if the folder is created successfully, otherwise False.
        
    Description:
        The function creates a folder in the specified path. If the path does not exist, it creates the path and then the folder.
    """

    try:
        os.makedirs(os.path.join(path, folder_name))
        
    except FileExistsError:
        print("The folder '{}' already exists in path '{}'.".format(folder_name, path))
        return False

    print("The folder '{}' has been created in path '{}'.".format(folder_name, path))
    
    return True
