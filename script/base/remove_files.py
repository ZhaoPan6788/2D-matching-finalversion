#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: remove_files.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-13
Description: A script to remove files in a list of specified folders or files.
"""

import os
import shutil

def remove_files(files_to_remove, source_path=None):
    """
    Arguments:
        files_to_remove (list): A list of folders or files to be removed.
    
    Returns:
        bool: True if all the specified files were removed successfully, False otherwise.
        source_path (str, optional): The source path. Defaults to None.

    Description:
        The function checks if the files or folders specified in the list exists. If not, it returns False with a warning message. 
        If all the files or folders exist, it removes them and returns True with a message indicating that the removal was successful.
    """

    if source_path is None:
        source_path = os.getcwd()

    for file in files_to_remove:
        file_target = os.path.join(source_path, file)
        if not os.path.exists(file_target):
            continue

        if os.path.isfile(file_target):
            os.remove(file_target)
        else:
            shutil.rmtree(file_target)
    
    print("Successfully removed all specified files.")
    return True
