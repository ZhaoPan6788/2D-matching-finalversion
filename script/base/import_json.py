#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: import_json.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: A script to import JSON file.
"""

import json

def import_json(file_path):
    """
    Arguments:
        file_path (str): The file path of the JSON file to be imported.
    
    Returns:
        dict: The dictionary object imported from the JSON file.
        
    Description:
        Import a JSON file and return the dictionary object.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print("File not found: {}".format(file_path))
        return None

if __name__ == "__main__":
    json_file_name = "../../config.json"
    json_data = import_json(json_file_name)
    if json_data:
        print("The json file exist.")
