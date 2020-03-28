# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
from graph_builder import model
from utils import *
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

#读取数据，label按倒数第一列进行拼接，拼接完成后，label是一个m*n的张量，m是样本数量，n是label总数
def npzload(ifn_data,ifn_label):
  d1 = np.load(ifn_data)
  feature = d1['data']
  l1 = np.load(ifn_label)
  label0,label1,label2 = l1['drug'],l1['manu'],l1['batch']
  return(feature,label0,label1,label2)

def get_feature_size(feature):
  size = len(feature[0])
  return(size)
  
def get_label_size(labels):
  size = []
  for i in range(len(labels)):
    nodes_count = len(labels[i][0])
    #nodes_count = len(nl[i][0])
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

def get_time():
  nower_time = time.time()
  return(nower_time)

def train_model(train_data,train_label,test_data,test_label,model_save_path,ofn,epochs,batch_size):
  matrices,label0,label1,label2 = npzload(train_data,train_label)
  test_matrices,test_label0,test_label1,test_label2 = npzload(test_data,test_label)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum,-1)
  labels = []
  labels.append(label0)
  labels.append(label1)
  labels.append(label2)

  labels_size = get_label_size(labels)
  matrices_size = get_feature_size(matrices)
  Model = model(feature = matrices, feature_size = matrices_size, label = labels, label_size = labels_size, lr = 1e-4)
  print('successful building Model!')
  minloss = 0.5

  #sample a batch
  N = len(matrices)
  all_batch = []
  for i in range(N):
    all_batch.append(i)
  all_batch = np.array(all_batch)
  N1 = len(test_matrices)
  test_batch = []
  for i in range(N1):
    test_batch.append(i)
  test_batch = np.array(test_batch)

  for itr in range(epochs):
    feed = {Model.x: matrices[all_batch]}
    feed[Model.y_0] = label0[all_batch]
    feed[Model.y_1] = label1[all_batch]
    feed[Model.y_2] = label2[all_batch]
    #train model
    loss,y_pred,y,logits,_ = Model.sess.run([Model.losses,Model.y_pred,Model.y_true, Model.logits, Model.train_op], feed)
    if((itr) % (epochs // 50) == 0):
      print("step:", (itr), "loss:",loss)
      em = eval(y,y_pred)
      print("exactly match rate:", em)
      if(minloss > loss):
        minloss = loss
        Model.save_json(model_save_path)
  feed1 = {Model.x: test_matrices[test_batch]}
  feed1[Model.y_0] = test_label0[test_batch]
  feed1[Model.y_1] = test_label1[test_batch]
  feed1[Model.y_2] = test_label2[test_batch]
  loss,y_pred,y = Model.sess.run([Model.losses,Model.y_pred,Model.y_true], feed1)
  log = open(ofn,'a')
  y_pred = np.around(y_pred, decimals=4)
  for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
      if(j == (len(y_pred[i])-1)):
        log.write(str(y_pred[i][j]))
      else:
        log.write(str(y_pred[i][j]) + ',')
    log.write('\t')
    for j in range(len(y[i])):
      if(j == (len(y[i])-1)):
        log.write(str(y[i][j]))
      else:
        log.write(str(y[i][j]) + ',')
    log.write('\n')
  log.close()

def main():
  train_model(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],1001,18)

if(__name__ == '__main__'):
  main()

