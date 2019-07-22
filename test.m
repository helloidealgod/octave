function test()
#定义样本
x = randn(28,28,1);
#x = randn(12,12,20);
#x = randn(5,5,40);
#定义标签
y = [0 0 0 0 0 0 0 0 1 0];
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
#前向计算
#计算卷积层1
for i = 1:20,
  c1(:,:,i) = max(0,convn(x,filter1(:,:,:,i),'valid') .+ bias1(i,1));
  [max_pool_1(:,:,i),max_index_1(:,:,i)] = maxPooling(c1(:,:,i));
end;
#计算卷积层2
for i = 1:40,
  c2(:,:,i) = max(0,convn(max_pool_1,filter2(:,:,:,i),'valid') .+ bias2(i,1));
  [max_pool_2(:,:,i),max_index_2(:,:,i)] = maxPooling(c2(:,:,i));
end;
#计算卷积层3
for i = 1:60,
  c3(:,:,i) = max(0,convn(max_pool_2,filter3(:,:,:,i),'valid') .+ bias3(i,1));
end;
z1 = w1*c3(:) + b1;
a1 = max(0,z1);
z2 = w2*a1 + b2;
p = softmax(z2);
#计算loss
loss = -log(p(9));
#fprintf("p = %f \n",p);
#printf("loss = %f \n",loss);
#循环,99%时停止
iterate = 0;
t=[0];
m=[loss];
plot = plot(t,m,'EraseMode','background','MarkerSize',5);
axis([0 50 -2.5 2.5]);
#打开网格
grid on;
while loss > 0.01,
#softmax层梯度
db2 = p - y';
dw2 = db2 * a1';
#全连接层梯度
db1 = w2' * db2 .* (grelu(z1));
dw1 = db1 * c3(:)';
#卷积层3梯度
dc3 = w1' * db1;
dc3 = reshape(dc3,3,3,60) .* grelu3(c3);
#db(60,1) = sum(dL)
#dw(3,3,40,60) = x(5,5,40) 卷积 dL(3,3,60) valid
#dx(5,5,40) = dL(3,3,60) 卷积 rot180(w(3,3,40,60)) full
dbias3 = zeros(60,1);
dfilter3 = zeros(3,3,40,60);
dx3 = zeros(5,5,40);
rfilter3 = rot90(rot90(filter3));
for i = 1 : 60,
  dbias3 = sum(sum(dc3(:,:,i)));
  for j = 1 : 40,
    dfilter3(:,:,j,i) = dfilter3(:,:,j,i) .+ convn(max_pool_2(:,:,j),dc3(:,:,i),'valid');
    dx3(:,:,j) = dx3(:,:,j) .+ convn(dc3(:,:,i),rfilter3(:,:,j,i),'full');
  end;
end;
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
#前向计算
#计算卷积层1
for i = 1:20,
  c1(:,:,i) = max(0,convn(x,filter1(:,:,:,i),'valid') .+ bias1(i,1));
  [max_pool_1(:,:,i),max_index_1(:,:,i)] = maxPooling(c1(:,:,i));
end;
#计算卷积层2
for i = 1:40,
  c2(:,:,i) = max(0,convn(max_pool_1,filter2(:,:,:,i),'valid') .+ bias2(i,1));
  [max_pool_2(:,:,i),max_index_2(:,:,i)] = maxPooling(c2(:,:,i));
end;
#计算卷积层3
for i = 1:60,
  c3(:,:,i) = max(0,convn(max_pool_2,filter3(:,:,:,i),'valid') .+ bias3(i,1));
end;
z1 = w1*c3(:) + b1;
a1 = max(0,z1);
z2 = w2*a1 + b2;
p = softmax(z2);
#计算loss
loss = -log(p(9));
iterate ++;
#fprintf("p = %f \n",p);
printf("iterate = %d,loss = %f \n",iterate,loss);

t=[t iterate];
m=[m loss];
set(plot,'XData',t,'YData',m);
drawnow;
#pause(0.01);
end;
end;