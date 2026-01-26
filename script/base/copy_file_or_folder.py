#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: copy_file_or_folder.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: Copy a list of files or folders from source path to target path.
"""

import os
import shutil

def copy_file_or_folder(file_or_folder_list, source_path=None, target_path=None):
    """
    Arguments:
        file_or_folder_list (list): A list of files or folders to be copied
        source_path (str, optional): The source path. Defaults to None.
        target_path (str, optional): The target path. Defaults to None.
    
    Returns:
        bool: True if the copy operation is successful, False otherwise.
        
    Description:
        Copy a list of files or folders from source path to target path. If source path is not specified, it will default to the current path. 
        The function will check if the files/folders and paths exist before copying, and will give a prompt if either does not exist.
    """

    if source_path is None:
        source_path = os.getcwd()
    if target_path is None:
        print("Target path is not specified.")
        return False

    if os.path.samefile(source_path, target_path):
        print("Source and target paths cannot be the same.")
        return False

    for file_or_folder in file_or_folder_list:
        src = os.path.join(source_path, file_or_folder)
        dst = os.path.join(target_path, file_or_folder)
        if not os.path.exists(src):
            print("Source {} does not exist.".format(src))
            return False
        if os.path.exists(dst):
            print("Target {} already exists.".format(dst))
            return False

    for file_or_folder in file_or_folder_list:
        src = os.path.join(source_path, file_or_folder)
        dst = os.path.join(target_path, file_or_folder)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            shutil.copytree(src, dst)

    return True

def copy_directory(src, dst):
    """
    Arguments:
        src (str): The source directory path.
        dst (str): The destination directory path.
        
    Returns:
        None
        
    Description:
        Copies the source directory to the destination directory.
    """
    try:
        shutil.copytree(src, dst)
        print("Directory copied from {} to {}".format(src, dst))

    except Exception as e:
        print("Directory copy failed: {}".format(e))

        return False

    return True

if __name__ == '__main__':
    copy_files = ['src', 'lib', 'config.json']
    if copy_file_or_folder(copy_files, '/home/chenzili/workspace/pic/ccp/iPM2D', '/home/chenzili'):
        print("Copying completed.")
