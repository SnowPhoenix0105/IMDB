#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author: SnowPhoenix


import os
import logging

class DefaultConfig:
    class PATH:
        TOP = os.getcwd()
        DATA = os.path.join(TOP, "data")
        DATA_PUBLIC = os.path.join(DATA, "public_data")
        DATA_TRAIN_CSV = os.path.join(DATA_PUBLIC, "train.csv")
        DATA_TEST_CSV = os.path.join(DATA_PUBLIC, "test_data.csv")
        DATA_TEMPLEMENT_CSV = os.path.join(DATA_PUBLIC, "submission.csv")
        DATA_RESULT_CSV = os.path.join(DATA, "submission.csv")

        CKPOINT = os.path.join(TOP, "ckpoint")
        IMAGE = os.path.join(TOP, "images")
        LOG = os.path.join(TOP, "log")

    class MODEL:
        BATCH_SIZE = 128
        CPU_BATCH_TIMES = 8
        VEC_LEN = 128
        PCA_DIM = 512

    class LOG:
        DEFAULT_LOG_DIR = "default.log"
        DEFAULT_HEAD = ''
        DEFAULT_MID = ''
        DEFAULT_NEED_CONSOLE = True

def auto_init():
    dir_to_ensure_exist = [
            DefaultConfig.PATH.CKPOINT, 
            DefaultConfig.PATH.IMAGE,
            DefaultConfig.PATH.LOG
        ]
    for d in dir_to_ensure_exist:
        if not os.path.exists(d):
            print("making dir:\t", d)
            os.makedirs(d)


auto_init()

if __name__ == "__main__":
    pass