# ONN4TCM
Ontology-aware Neural Network for Traditional Chinese Medicine!

This program was designed to perform fast and accurate source tracking of TCM preparations.

## Install
Dowload the zip archive from this [repository][1], and unzip the archive.

## Function
The program can be used for source tracking of TCM preparations. Moreover, it can also generate ROC curves diagram. All analyses could be realized by run the "scripts/run.sh".

## Dependencies
-[[tensorflow-gpu-1.14.0]][2]  
-[python-3.7]  
-[matplotlib-3.1.0]  
-[numpy-1.16.4]  

## Usage
Firstly, change into the directory where you save the ONN4TCM. Secondly, fix the path in "scripts/run.sh" Finally, run the "scripts/run.sh". Type,
```shell
sh scripts/run.sh >/dev/null 
```
If you want to run the "scripts/run.sh" in the background, type,
```shell
nohup sh scripts/run.sh >/path/to/log.txt &
```

## Author
   Name   |      Email      |      Organization
:--------:|-----------------|--------------------------------------------------------------------------------------------------------------------------------
Hugo Zha |hugozha@hust.deu.cn|Ph.D. Candidate, School of Life Science and Technology, Huazhong University of Science & Technology
Kang Ning|ningkang@hust.edu.cn|Professor, School of Life Science and Technology, Huazhong University of Science & Technology

[1]:https://github.com/HUST-NingKang-Lab/ONN4TCM
[2]:https://pypi.org/project/tensorflow-gpu/1.14.0/
