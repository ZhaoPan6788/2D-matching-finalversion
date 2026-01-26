#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: parse_command_line_input.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2023-02-03
Description: A script to parse command line input into key-value pairs and store them in a dictionary.
"""

import sys

def parse_input(argv):
    """
    Arguments:
        argv (list of str): A list of command line arguments.

    Returns:
        dict: A dictionary of key-value pairs parsed from command line input.
        
    Description:
        This function parses command line input into key-value pairs and stores them in a dictionary.
    """
    d = {}
    i = 1
    while i < len(argv):
        if argv[i].startswith("-"):
            key = argv[i][1:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                try:
                    val = int(argv[i + 1])
                except ValueError:
                    try:
                        val = float(argv[i + 1])
                    except ValueError:
                        if argv[i + 1].lower() == "true":
                            val = True
                        elif argv[i + 1].lower() == "false":
                            val = False
                        else:
                            val = argv[i + 1]
                i += 1
            else:
                val = True
            d[key] = val
        i += 1
    return d

if __name__ == "__main__":
    input_dict = parse_input(sys.argv)
    if input_dict:
        print("Parsed command line input:")
        for key, val in input_dict.items():
            print("\t{}: {}".format(key, val))
