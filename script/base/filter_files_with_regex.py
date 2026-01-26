#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: filter_files_with_regex.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-04
Description: Filter files and folders in a specified path based on a given regular expression.
"""

import os
import re

def filter_files_with_regex(regex, path="."):
    """
    Arguments:
        regex (str): A regular expression string to match the file/folder name.
        path (str, optional): The path to search files and folders. Default is ".".
    
    Returns:
        list: A list of all the matched file/folder names.
        
    Description:
        This function filters files and folders in the specified path based on the given regular expression. 
        If the path does not exist, an empty list will be returned.
    """
    if not os.path.exists(path):
        return []

    files = os.listdir(path)
    matched_files = [f for f in files if re.search(regex, f)]

    return matched_files


if __name__ == '__main__':
    matched_files = filter_files_with_regex('.+[.]cmake', '/home/chenzili/workspace/pic/ccp/iPM2D')

    if matched_files:
        print("Matched files/folders:")
        for file in matched_files:
            print("- {}".format(file))
