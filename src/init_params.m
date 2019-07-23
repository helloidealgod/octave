function [filter1,bias1,vfilter1,vbias1,filter2,bias2,vfilter2,vbias2,filter3,bias3,vfilter3,vbias3,w1,b1,vw1,vb1,w2,b2,vw2,vb2] = init_params()
#定义卷积层1
filter1 = 0.01 * randn(5,5,1,20);
bias1 = 0.01 * rand(20,1);
vfilter1 = zeros(5,5,1,20);
vbias1 = zeros(20,1);
#定义卷积层2
filter2 = 0.01 * randn(3,3,20,40);
bias2 = 0.01 * rand(40,1);
vfilter2 = zeros(3,3,20,40);
vbias2 = zeros(40,1);
#定义卷积层3
filter3 = 0.01 * randn(3,3,40,60);
bias3 = 0.01 * rand(60,1);
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
end;