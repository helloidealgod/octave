# x样本，n行d列，
# d:特征维数
# n:样本个数
# x:sample set
#生成n行d列矩阵
x = rand(n,d);
# 样本，标志第几个样本是属于哪个人的特征
label = rand(n,1);
#mean(x) 按列求平均 
x = x - mean(a);
