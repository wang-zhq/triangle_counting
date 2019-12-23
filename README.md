# triangle_counting
关于对图数据进行三角形计数的CPU程序

## 编译命令
```
g++ -fopenmp -o countri countri.cpp -O3 -std=c++17
```

## 运行命令
```
./countri path/dataname.xxx
```

## 命题来源
请参见 https://www.datafountain.cn/competitions/349/datasets

## 数据来源
数据文件格式仅限32位整型二进制文件
+ 来源一：https://snap.stanford.edu/data/ 相应txt格式文件可下载本人 txt2bin 工具进行转换
+ 来源二：用本人 kron_generator 工具生成二进制数据集

## 算法步骤
+ 将图数据载入为数组
+ 遍历数据找到最大值，即为顶点个数
+ 统计顶点总度数，并按度数大小排序后生成新的编号，对原数据进行替换（没有该步骤，不影响结果正确性，但会大幅增加计算时间）
+ 整理数据，对数据进行排序，将数据规整为上三角矩阵
+ 对数据进行去重、归并作，建立行区间索引表
+ 根据计算原理与行区间索引表对排序后的图数据进行计算，即得到三角形计数


## 评测说明
本程序计算s24数据集在8CPU核下能够跑进40s，并且还有改进空间

## 其它
本文档将持续完善
