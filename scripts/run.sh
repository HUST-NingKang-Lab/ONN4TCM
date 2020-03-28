#!/bin/bash
# -*- coding: utf-8 -*-

#set path
#path=/abs/path/to/ONN4TCM/archive/
#example as follows:
path=/data1/zhayuguo/ST4TCM/ONN4TCM
cd $path
'''
cmdpy=${path}/scripts/train_model.py
rm -f result/{ITS2,TRNL}/split*/result?.txt

for i in result/{ITS2,TRNL}/split* ; do
  cd $i
  n=`echo $i | cut -c 18`
  echo $n
  python $cmdpy ./train${n}_data.npz ./train${n}_label.npz ./test${n}_data.npz ./test${n}_label.npz ./model.json ./result${n}.txt
  wait
  cd ../../..
done

rm -f result/{ITS2,TRNL}/*txt
cat result/ITS2/split*/result?.txt >> result/ITS2/its2_result.txt
cat result/TRNL/split*/result?.txt >> result/TRNL/trnl_result.txt

#compute fpr, tpr, accuracy, precision, recall, f1score
python scripts/evaluate.py result/ITS2/its2_result.txt result/ITS2/its2_result_drug_roc.txt result/ITS2/its2_result_manu_roc.txt result/ITS2/its2_result_drug.csv result/ITS2/its2_result_manu.csv result/ITS2/its2_result_batch_roc.txt result/ITS2/its2_result_batch.csv
python scripts/evaluate.py result/TRNL/trnl_result.txt result/TRNL/trnl_result_drug_roc.txt result/TRNL/trnl_result_manu_roc.txt result/TRNL/trnl_result_drug.csv result/TRNL/trnl_result_manu.csv result/TRNL/trnl_result_batch_roc.txt result/TRNL/trnl_result_batch.csv
'''
#draw roc curve
python scripts/draw_roc.py result/ITS2/its2_result_drug_roc.txt result/TRNL/trnl_result_drug_roc.txt result/figure/ONN4TCM-ROC-preparation.png ONN4TCM-ROC-preparation
python scripts/draw_roc.py result/ITS2/its2_result_manu_roc.txt result/TRNL/trnl_result_manu_roc.txt result/figure/ONN4TCM-ROC-manufacturer.png ONN4TCM-ROC-manufacturer
python scripts/draw_roc.py result/ITS2/its2_result_batch_roc.txt result/TRNL/trnl_result_batch_roc.txt result/figure/ONN4TCM-ROC-batch.png ONN4TCM-ROC-batch
cd ~
