# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:21:41 2019

@author: anonymous
"""
import numpy as np


def cost_func(a, b):
    return np.linalg.norm(a - b)