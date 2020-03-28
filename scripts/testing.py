# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
import math
import json
import time
import pickle
from graph_builder import model
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

#读取数据，label按倒数第一列进行拼接，拼接完成后，label是一个m*n的张量，m是样本数量，n是label总数
def npzload(ifn_data,ifn_label):
  d1 = np.load(ifn_data)
  feature = d1['data']
  l1 = np.load(ifn_label)
  label0,label1 = l1['drug'],l1['manu']
  return(feature,label0,label1)

def npzload1(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  return(feature)

def get_feature_size(feature):
  size = len(feature[0])
  return(size)
  
def get_label_size(labels):
  size = []
  for i in range(len(labels)):
    nodes_count = len(labels[i][0])
    size.append(nodes_count)
  return(size)

#计算一个样本是否预测得exact match
def eval_labels(y_true, y_pred):
  for i in range(len(y_true)):
    if float(y_true[i]) == 1 and float(y_pred[i]) < 0.5:
      return 0
    if float(y_true[i]) == 0 and float(y_pred[i]) >= 0.5:
      return 0
  return 1

#计算exact match rate
def eval(y_true, y_pred):
  cnt, s = 0.0, 0.0
  for i in range(len(y_true)):
    cnt += 1
    s += eval_labels(y_true[i], y_pred[i])
  if cnt == 0:
    return 0
  return(s/cnt)


def test_model(fn_data,fn_label,Model):
  matrices,label0,label1 = npzload(fn_data,fn_label)
  #matrices = npzload1(fn)
  print('successful loading data!')
  
  #sample a batch
  N = len(matrices)
  all_batch = []
  for i in range(N):
    all_batch.append(i)
  all_batch = np.array(all_batch)

  #feed the feature to our model
  feed = {Model.x: matrices[all_batch]}
  feed[Model.y_0] = label0[all_batch]
  feed[Model.y_1] = label1[all_batch]
  #train model
  y_pred,y = Model.sess.run([Model.y_pred,Model.y_true], feed)

def main():
  dddd = time.time()
  print("start time is: ",dddd)
  #sys.argv[3] is tree
  matrices,label0,label1 = npzload(sys.argv[1],sys.argv[2])
  dddd = time.time()
  print("read data time is: ",dddd)

  labels = []
  labels.append(label0)
  labels.append(label1)
  labels_size = get_label_size(labels)
  matrices_size = get_feature_size(matrices)
  Model = model(feature_size = matrices_size, label_size = labels_size)
  #sys.argv[2] is trained model
  Model.load_json(sys.argv[3])
  dddd = time.time()
  print("load model time is: ",dddd)

  #sys.argv[1] is the testing data
  test_model(sys.argv[1],sys.argv[2],Model)
  dddd = time.time()
  print("end time is: ",dddd)
if(__name__ == '__main__'):
  main()

