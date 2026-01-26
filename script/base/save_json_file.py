#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: save_json_file.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-06
Description: A script to save json file with specified path and file name.
"""

import os
import json

def save_json_file(file_path, file_name, data):
    """
    Arguments:
        file_path (str): The path to save the json file
        file_name (str): The name of the json file
        data (dict): The data to be saved in json format
    
    Returns:
        bool: Return True if the file is saved successfully, otherwise False.
        
    Description:
        The function checks if the path and file exist, if the path does not exist,
        create the path. If the file exists, replace it. Finally, save the data in json format.
    """

    full_path = os.path.join(file_path, file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    try:
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=4)
            print("The json file is saved at {}".format(full_path))
            return True
    except Exception as e:
        print("Failed to save the json file: {}".format(e))
        return False
