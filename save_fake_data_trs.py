#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 02:33:18 2023

@author: walidajalil
"""

import numpy as np

shape_train = (48410, 300, 58, 3)
shape_val = (19859, 300, 58, 3)


data_train = np.random.uniform(low=0.0, high=1.0, size=shape_train)
data_val = np.random.uniform(low=0.0, high=1.0, size=shape_val)
np.save('train_single_person.npy', data_train)
np.save('val_single_person.npy', data_val)