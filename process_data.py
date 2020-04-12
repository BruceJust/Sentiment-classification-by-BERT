# -*- coding: utf-8 -*-
# @Time    : 2020/4/4 21:03
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : process_data
# @Project: sentiment

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 将数据随机拆分为train，test
def prepare_data():
    df = pd.read_csv('./data/waimai_10k.csv')
    x = df[['review', 'label']]
    y = np.arange(0, len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_train.to_csv('data/train.csv',  index=False, header=False)
    x_test.to_csv('data/dev.csv',  index=False, header=False)

prepare_data()