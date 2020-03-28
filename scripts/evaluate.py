#!/usr/bin/env python3
import numpy as np
import sys

def read_result(ifn):
  with open(ifn,'r') as f:
    pred,true = [],[]
    for line in f.readlines():
      line = line.strip()
      tp = line.split('\t')[0]
      tt = line.split('\t')[1]
      tpred = tp.split(',')
      ttrue = tt.split(',')
      for i in range(len(tpred)):
        tpred[i] = float(tpred[i])
        ttrue[i] = float(ttrue[i])
      pred.append(tpred)
      true.append(ttrue)
    pred1 = np.array(pred)
    true1 = np.array(true)
    return(pred1,true1)

def evalres(pred,true,ofn1,ofn2,ofn3,ofn4,ofn5,ofn6):
  sn = len(pred)
  pred_drug, pred_manu,pred_batch = pred[:,0:4], pred[:,4:6], pred[:,6:9]
  true_drug, true_manu,true_batch = true[:,0:4], true[:,4:6], true[:,6:9]

  #drug
  fpr_drug,tpr_drug = [],[]
  f3 = open(ofn3,'a')
  f3.write('threshold' + '\t' + 'accuracy' + '\t' + 'precision' + '\t' + 'recall' + '\t' + 'f1-score' + '\n')
  for n in range(102):
    th = n/100
    TTP,TTN,TFP,TFN = 0,0,0,0
    for i in range(sn):
      TP,TN,FP,FN = 0,0,0,0
      for j in range(len(pred_drug[i])):
        if(pred_drug[i][j] >= th and true_drug[i][j] == 1):
          TP += 1
        if(pred_drug[i][j] < th and true_drug[i][j] == 1):
          FN += 1
        if(pred_drug[i][j] >= th and true_drug[i][j] == 0):
          FP += 1
        if(pred_drug[i][j] < th and true_drug[i][j] == 0):
          TN += 1
      TTP += TP
      TTN += TN
      TFP += FP
      TFN += FN
    TPR = TTP/sn
    FPR = TFP/(3*sn)
    fpr_drug.append(FPR)
    tpr_drug.append(TPR)
    accuracy = (TTP + TTN)/(TTP + TTN + TFP + TFN)
    pr = precision = TTP/(TTP+TFP+0.00001)
    rc = recall = TTP/(TTP+TFN)
    f1score = 2*pr*rc/(pr + rc+0.00001)
    f3.write(str(th) + '\t' + str(accuracy) + '\t' + str(pr) + '\t' + str(rc) + '\t' + str(f1score) + '\n')
  f3.close()
  f1 = open(ofn1,'a')
  for i in range(len(fpr_drug)):
    f1.write(str(fpr_drug[i]) + '\t' + str(tpr_drug[i]) + '\n')
  f1.close()

  #manu
  fpr_manu,tpr_manu = [],[]
  f4 = open(ofn4,'a')
  f4.write('threshold' + '\t' + 'accuracy' + '\t' + 'precision' + '\t' + 'recall' + '\t' + 'f1-score' + '\n')

  for n in range(102):
    th = n/100
    TTP,TTN,TFP,TFN = 0,0,0,0
    for i in range(sn):
      TP,TN,FP,FN = 0,0,0,0
      for j in range(len(pred_manu[i])):
        if(pred_manu[i][j] >= th and true_manu[i][j] == 1):
          TP += 1
        if(pred_manu[i][j] < th and true_manu[i][j] == 1):
          FN += 1
        if(pred_manu[i][j] >= th and true_manu[i][j] == 0):
          FP += 1
        if(pred_manu[i][j] < th and true_manu[i][j] == 0):
          TN += 1
      TTP += TP
      TTN += TN
      TFP += FP
      TFN += FN
    TPR = TTP/sn
    FPR = TFP/sn
    fpr_manu.append(FPR)
    tpr_manu.append(TPR)
    accuracy = (TTP + TTN)/(TTP + TTN + TFP + TFN)
    pr = precision = TTP/(TTP+TFP+0.00001)
    rc = recall = TTP/(TTP+TFN)
    f1score = 2*pr*rc/(pr + rc+0.00001)
    f4.write(str(th) + '\t' + str(accuracy) + '\t' + str(pr) + '\t' + str(rc) + '\t' + str(f1score) + '\n')
  f4.close()

  f2 = open(ofn2,'a')
  for i in range(len(fpr_manu)):
    f2.write(str(fpr_manu[i]) + '\t' + str(tpr_manu[i]) + '\n')
  f2.close()

  #batch
  fpr_batch,tpr_batch = [],[]
  f6 = open(ofn6,'a')
  f6.write('threshold' + '\t' + 'accuracy' + '\t' + 'precision' + '\t' + 'recall' + '\t' + 'f1-score' + '\n')

  for n in range(102):
    th = n/100
    TTP,TTN,TFP,TFN = 0,0,0,0
    for i in range(sn):
      TP,TN,FP,FN = 0,0,0,0
      for j in range(len(pred_batch[i])):
        if(pred_batch[i][j] >= th and true_batch[i][j] == 1):
          TP += 1
        if(pred_batch[i][j] < th and true_batch[i][j] == 1):
          FN += 1
        if(pred_batch[i][j] >= th and true_batch[i][j] == 0):
          FP += 1
        if(pred_batch[i][j] < th and true_batch[i][j] == 0):
          TN += 1
      TTP += TP
      TTN += TN
      TFP += FP
      TFN += FN
    TPR = TTP/sn
    FPR = TFP/(2*sn)
    fpr_batch.append(FPR)
    tpr_batch.append(TPR)
    accuracy = (TTP + TTN)/(TTP + TTN + TFP + TFN)
    pr = precision = TTP/(TTP+TFP+0.00001)
    rc = recall = TTP/(TTP+TFN)
    f1score = 2*pr*rc/(pr + rc+0.00001)
    f6.write(str(th) + '\t' + str(accuracy) + '\t' + str(pr) + '\t' + str(rc) + '\t' + str(f1score) + '\n')
  f6.close()

  f5 = open(ofn5,'a')
  for i in range(len(fpr_batch)):
    f5.write(str(fpr_batch[i]) + '\t' + str(tpr_batch[i]) + '\n')
  f5.close()


def main():
  pred,true = read_result(sys.argv[1])
  evalres(pred,true,sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])
  
main()
