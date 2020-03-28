#!/usr/bin/env python3
import numpy as np
import scipy.stats
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import auc


def get_ftpr(ifn):
  fpr,tpr = [],[]
  with open(ifn,'r') as f:
    for line in f.readlines():
      line = line.strip()
      fpr.append(float(line.split('\t')[0]))
      tpr.append(float(line.split('\t')[1]))
  return(fpr,tpr)

def draw(ifn1,ifn2,ofn,title):
  fpr1,tpr1 = get_ftpr(ifn1)
  fpr2,tpr2 = get_ftpr(ifn2)

  roc_auc1 = auc(fpr1, tpr1)
  print(roc_auc1)
  roc_auc2 = auc(fpr2, tpr2)
  print(roc_auc2)

  plt.figure()
  lw = 2
  plt.figure(figsize=(10,10))
  plt.plot(fpr1,tpr1,color='darkorange',lw=lw,label='ONN4TCM-ITS2 ROC curve(area = %0.4f)' % roc_auc1)
  plt.plot(fpr2,tpr2,color='darkblue',lw=lw,label='ONN4TCM-TRNL ROC curve(area = %0.4f)' % roc_auc2)
  #plt(fpr1,tpr1,color='darkorange',lw=lw,label='JSD-ITS2_FS ROC curve(area = %0.2f)' % roc_auc1)
  #plt.plot(fpr2,tpr2,color='blue',lw=lw,label='JSD-TRNL_FS ROC curve(area = %0.2f)' % roc_auc2)
  #plt.plot(fpr3,tpr3,color='red',lw=lw,label='JSD-CHEMICAL_FS ROC curve(area = %0.2f)' % roc_auc3)
  #plt.plot(fpr4,tpr4,color='green',lw=lw,label='JSD-MIX_FS ROC curve(area = %0.2f)' % roc_auc4)

  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xticks(fontsize = 12)
  plt.yticks(fontsize = 12)
  plt.xlabel('False Positive Rate',fontsize=16)
  plt.ylabel('True Positive Rate',fontsize=16)
  #plt.title('Receiver operating characteristic of JSD')
  plt.title(title,fontsize=24)
  plt.legend(loc="lower right",fontsize=12)
  #plt.savefig("Receiver operating characteristic of JSD.png",dpi=600)
  plt.savefig(ofn,dpi=600)

draw(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
