#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: integer_grouping.py
Author: Zili Chen
Email: chen__zili@163.com
Date: 2022-10-11
Description: A function that groups a list of integers into multiple groups with a maximum capacity.
"""

def integer_grouping(int_list, max_capacity):
    """
    Arguments:
        int_list (list): A list of integers.
        max_capacity (int): The maximum capacity of each group.
    
    Returns:
        list: A list of groups, where each group is a list of integers.
        list: A list of tuples, where each tuple contains the index of the integer in the original list and the index of the group it belongs to.
        
    Description:
        This function groups a list of integers into multiple groups with a maximum capacity. If a group contains multiple integers, the sum of these integers must be less than or equal to the maximum capacity. If a group contains only one integer, the integer value can be greater than the maximum capacity. The function tries to minimize the number of groups.
    """

    groups = []
    int_index = []
    current_group = []
    current_sum = 0
    group_index = 0

    for i, num in enumerate(int_list):
        if num > max_capacity:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_sum = 0
                group_index += 1

            groups.append([num])
            int_index.append((i, group_index))
            group_index += 1
        else:
            if current_sum + num <= max_capacity:
                current_group.append(num)
                current_sum += num
                int_index.append((i, group_index))
            else:
                groups.append(current_group)
                current_group = [num]
                current_sum = num
                group_index += 1
                int_index.append((i, group_index))

    if current_group:
        groups.append(current_group)

    index_groups = [[] for i in range(len(groups))]
    for i in int_index:
        index_groups[i[1]].append(i[0])

    return groups, index_groups, int_index

if __name__ == '__main__':
    int_list = [1, 2, 4, 8, 16, 2, 4, 8, 16, 32, 4, 8, 16, 32, 64]
    max_capacity = 48
    result, index_groups, index = integer_grouping(int_list, max_capacity)
    
    print("Groups:")
    for group in result:
        print(group)
    
    print("Index Groups:")
    for igrp in index_groups:
        print(igrp)

    print("Index:")
    for i in index:
        print("Integer {} belongs to Group {}".format(i[0], i[1]))
