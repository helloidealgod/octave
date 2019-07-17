function test()
#定义样本
x = randn(540,1);
#定义标签
y = [1 0 0 0 0 0 0 0 0 0];
#定义卷积层3
filter3 = randn(3,3,40,60);
bias3 = rand(60,1);
vfilter3 = zeros(3,3,40,60);
vbias3 = zeros(60,1);
#定义全连接层
w1 = 0.01 * randn(160,540);
b1 = 0.01 * randn(160,1);
vw1 = zeros(160,540);
vb1 = zeros(160,1);
#softmax层
w2 = 0.01 * randn(10,160);
b2 = 0.01 * randn(10,1);
vw2 = zeros(10,160);
vb2 = zeros(10,1);
#前向计算
z1 = w1*x + b1;
a1 = max(0,z1);
z2 = w2*a1 + b2;
p = softmax(z2);
#计算loss
loss = -log(p(1));
#循环,99%时停止
while loss > 0.01,
#softmax层梯度
db2 = p - y';
dw2 = db2 * a1';
#全连接层梯度
db1 = w2' * db2 .* (grelu(z1));
dw1 = db1 * x';
#动量梯度下降法
vw2 = 0.9 * vw2 + 0.1 * dw2;
vb2 = 0.9 * vb2 + 0.1 * db2;
w2 = w2 - 0.1 * vw2;
b2 = b2 - 0.1 * vb2;
#更新参数
vw1 = 0.9 * vw1 + 0.1 * dw1;
vb1 = 0.9 * vb1 + 0.1 * db1;
w1 = w1 - 0.1 * vw1;
b1 = b1 - 0.1 * vb1;
#前向计算
z1 = w1*x + b1;
a1 = max(0,z1);
z2 = w2*a1 + b2;
p = softmax(z2);
loss = -log(p(1));
fprintf("loss = %f \n",loss);
end;
end;