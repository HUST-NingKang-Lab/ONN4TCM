# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import math
import sys

def split(ifn_data,ifn_label,splitn):
  d1 = np.load(ifn_data)
  l1 = np.load(ifn_label)
  matrix = d1['data']
  label0 = l1['drug']
  label1 = l1['manu']
  label2 = l1['batch']
  #打乱数据顺序
  samplenum = len(matrix)
  index = [i for i in range(samplenum)]
  np.random.shuffle(index)
  mylst = np.array(index)
  np.savez('samplelst',data=mylst)
  matrix = matrix[index]
  label0 = label0[index]
  label1 = label1[index]
  label2 = label2[index]

  foldsize = math.floor(samplenum / splitn)
  Feature,L0,L1,L2,L3,L4,L5 = [],[],[],[],[],[],[]
  for i in range(splitn):
    start = i*foldsize
    end = (i+1)*foldsize
    feature = matrix[start:end,:]
    #print(len(feature))
    #print(feature.shape)
    l0= label0[start:end,:]
    l1= label1[start:end,:]
    l2= label2[start:end,:]
    Feature.append(feature)
    L0.append(l0)
    L1.append(l1)
    L2.append(l2)
  Feature = np.array(Feature)
  L0 = np.array(L0)
  L1 = np.array(L1)
  L2 = np.array(L2)
  print(Feature.shape)
  print(L0.shape)
  for i in range(splitn):
    savepath = ''
    trainlist = []
    for j in range(1,splitn):
      trainlist.append(i-j)
    #print(i)
    #print(trainlist)
    #print('---------------------------------------------------')
    tmp_matrix,tmp_L0,tmp_L1,tmp_L2 = Feature[trainlist[0]],L0[trainlist[0]],L1[trainlist[0]],L2[trainlist[0]]
    for k in range(1,len(trainlist)):
      tmp_matrix = np.concatenate((tmp_matrix,Feature[trainlist[k]]),axis=0)
      tmp_L0 = np.concatenate((tmp_L0,L0[trainlist[k]]),axis=0)
      tmp_L1 = np.concatenate((tmp_L1,L1[trainlist[k]]),axis=0)
      tmp_L2 = np.concatenate((tmp_L2,L2[trainlist[k]]),axis=0)

    matrix_train = tmp_matrix
    L0_train,L1_train,L2_train = tmp_L0,tmp_L1,tmp_L2
    matrix_test = Feature[i]
    L0_test,L1_test,L2_test = L0[i],L1[i],L2[i]
    trainfn_data = savepath + 'split' + str(i) + '/train' + str(i) + '_data.npz'
    trainfn_label = savepath + 'split' + str(i) + '/train' + str(i) + '_label.npz'
    testfn_data = savepath + 'split' + str(i) + '/test' + str(i) + '_data.npz'
    testfn_label = savepath + 'split' + str(i) + '/test' + str(i) + '_label.npz'
    np.savez(trainfn_data, data=matrix_train)
    np.savez(testfn_data, data=matrix_test)
    np.savez(trainfn_label, drug=L0_train, manu=L1_train, batch=L2_train)
    np.savez(testfn_label, drug=L0_test, manu=L1_test, batch=L2_test)
  return 0

split(sys.argv[1],sys.argv[2],4)
