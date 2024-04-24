import os
import numpy as np
import csv
import random

random.seed(20213385) #设置学号

def process_data1(inpath, outpath, nline):
    pos_data = list()
    neg_data = list()
    with open(inpath) as fin:
        for line in fin:
            line = line.strip().strip(".")
            tokens = line.split(", ")
            if tokens[-1].strip() == "<=50K":
                neg_data.append(line)
            elif tokens[-1].strip() == ">50K":
                pos_data.append(line)
    with open(outpath + ".csv", "w") as fout:
        csv_writer = csv.writer(fout)
        random.shuffle(pos_data)
        random.shuffle(neg_data)
        print(len(pos_data))
        print(len(neg_data))
        data = pos_data[:nline] + neg_data[:nline]
        for step, line in enumerate(data):
            tokens =line.split(", ")
            csv_writer.writerow(tokens)

def process_data2(inpath, outpath, nline):
    pos_data = list()
    neg_data = list()
    with open(inpath) as fin:
        for line in fin:
            line = line.strip().strip(".")
            tokens = line.split(", ")
            if tokens[-1].strip() == "<=50K":
                neg_data.append(line)
            elif tokens[-1].strip() == ">50K":
                pos_data.append(line)
    with open(outpath + ".csv", "w") as fout:
        csv_writer = csv.writer(fout)
        random.shuffle(pos_data)
        random.shuffle(neg_data)
        print(len(pos_data))
        print(len(neg_data))
        data = pos_data[:nline] + neg_data[:nline]
        for step, line in enumerate(data):
            tokens =line.split(", ")
            csv_writer.writerow(tokens)

process_data1("Income.data", "deal", 7841)
process_data2("Income.test", "answer", 3000)
