function test()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据读取%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('start reading file ... \n');
images = loadMNISTImages("../resource/t10k-images.idx3-ubyte");
labels = loadMNISTLabels("../resource/t10k-labels.idx1-ubyte");
fprintf('file read done \n');
[m,n] = size(images);
x = zeros(28,28,n);
fprintf('start loading data ...\n');
for i = 1 : n,
  x(:,:,i) = reshape(images(:,i),28,28);
end;
fprintf('data load done \n');
clear images;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%数据读取%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#定义标签
#y = [0 0 0 0 0 0 0 0 1 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#前向计算
#计算卷积层1
for i = 1:20,
  c1(:,:,i) = max(0,convn(x(:,:,1),filter1(:,:,:,i),'valid') .+ bias1(i,1));
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
loss = -log(p(labels(1)));
#[c1,max_pool_1,max_index_1,c2,max_pool_2,max_index_2,c3,z1,a1,z2,p,loss] = forward(x(:,:,1),filter1,filter2,filter3,w1,w2,bias1,bias2,bias3,b1,b2,labels(1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#循环,99%时停止
iterate = 0;
t=[0];
m=[loss];
plot = plot(t,m,'EraseMode','background','MarkerSize',5);
axis([0 300 -2.5 2.5]);
#打开网格
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while loss > 0.01,
#softmax层梯度
db2 = p;
db2(labels(1)) = db2(labels(1)) - 1;
#db2 = p - y';
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
#卷积层2梯度
kw = [1 1;1 1];
dc2 = zeros(10,10,40);
for i = 1 : 40,
    dc2(:,:,i) = kron(dx3(:,:,i),kw);
end;
dc2 = dc2 .* max_index_2;

dfilter2 = zeros(3,3,20,40);
dx2 = zeros(12,12,20);
rfilter2 = rot90(rot90(filter2));
for i = 1 : 40,
    dbias2 = sum(sum(dc2(:,:,i)));
    for j = 1 : 20,
    dfilter2(:,:,j,i) = dfilter2(:,:,j,i) .+ convn(max_pool_1(:,:,j),dc2(:,:,i),'valid');
    dx2(:,:,j) = dx2(:,:,j) .+ convn(dc2(:,:,i),rfilter2(:,:,j,i),'full');
    end;
end;
#卷积层2
dc1 = zeros(24,24,20);
for i = 1 : 20,
    dc1(:,:,i) = kron(dx2(:,:,i),kw);
end;
dc1 = dc1 .* max_index_1;
dfilter1 = zeros(5,5,1,20);
for i = 1 : 20,
    dbias1 = sum(sum(dc1(:,:,i)));
    for j = 1 : 1,
    dfilter1(:,:,j,i) = dfilter1(:,:,j,i) .+ convn(x(:,:,j),dc1(:,:,i),'valid');
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
#计算卷积层1
for i = 1:20,
  c1(:,:,i) = max(0,convn(x(:,:,1),filter1(:,:,:,i),'valid') .+ bias1(i,1));
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
loss = -log(p(labels(1)));
iterate ++;
#fprintf("p = %f \n",p);
printf("iterate = %d,loss = %f \n",iterate,loss);

t=[t iterate];
m=[m loss];
set(plot,'XData',t,'YData',m);
drawnow;
#pause(0.01);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;