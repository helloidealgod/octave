addpath('./math');
addpath('./data');
#读取数据
[images,labels] = loadData();
#参数初始化
[filter1,bias1,vfilter1,vbias1,filter2,bias2,vfilter2,vbias2,filter3,bias3,vfilter3,vbias3,w1,b1,vw1,vb1,w2,b2,vw2,vb2] = init_params();
#前向计算
[c1,max_pool_1,max_index_1,c2,max_pool_2,max_index_2,c3,z1,a1,z2,p,loss] = ...
forward(images,filter1,filter2,filter3,w1,w2,bias1,bias2,bias3,b1,b2,labels(1));
fprintf('loss = %f \n',loss);
iterate = 0;
t=[0];
m=[loss];
plot = plot(t,m,'EraseMode','background','MarkerSize',5);
axis([0 300 -2.5 2.5]);
#打开网格
grid on;
while loss > 0.01,
#后向计算
[dfilter1,dfilter2,dfilter3,dw1,dw2,dbias1,dbias2,dbias3,db1,db2] = ...
backward(images,p,labels(1),max_pool_1,max_index_1,filter2,max_pool_2,max_index_2,filter3,c3,w1,w2,z1,a1);

#动量梯度下降法 softmax层更新参数
vw2 = 0.9 * vw2 + 0.1 * dw2;
vb2 = 0.9 * vb2 + 0.1 * db2;
w2 = w2 - 0.1 * vw2;
b2 = b2 - 0.1 * vb2;
#动量梯度下降法 全连接层更新参数
vw1 = 0.9 * vw1 + 0.1 * dw1;
vb1 = 0.9 * vb1 + 0.1 * db1;
w1 = w1 - 0.1 * vw1;
b1 = b1 - 0.1 * vb1;
#动量梯度下降法 卷积层3更新参数
vfilter3 = 0.9 * vfilter3 + 0.1 * dfilter3;
vbias3 = 0.9 * vbias3 + 0.1 * dbias3;
filter3 = filter3 - 0.01 * vfilter3;
bias3 = bias3 - 0.01 * vbias3;
#动量梯度下降法 卷积层2更新参数
vfilter2 = 0.9 * vfilter2 + 0.1 * dfilter2;
vbias2 = 0.9 * vbias2 + 0.1 * dbias2;
filter2 = filter2 - 0.01 * vfilter2;
bias2 = bias2 - 0.01 * vbias2;
#动量梯度下降法 卷积层1更新参数
vfilter1 = 0.9 * vfilter1 + 0.1 * dfilter1;
vbias1 = 0.9 * vbias1 + 0.1 * dbias1;
filter1 = filter1 - 0.01 * vfilter1;
bias1 = bias1 - 0.01 * vbias1;
#前向计算
[c1,max_pool_1,max_index_1,c2,max_pool_2,max_index_2,c3,z1,a1,z2,p,loss] = ...
forward(images,filter1,filter2,filter3,w1,w2,bias1,bias2,bias3,b1,b2,labels(1));
iterate ++;
printf("iterate = %d,loss = %f \n",iterate,loss);
t=[t iterate];
m=[m loss];
set(plot,'XData',t,'YData',m);
drawnow;
end;