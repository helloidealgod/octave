function [dconv_out_3,dfilter_3] = main()
fprintf('start reading file ... \n');
images = loadMNISTImages("t10k-images.idx3-ubyte");
labels = loadMNISTLabels("t10k-labels.idx1-ubyte");
fprintf('file read done \n');
#fprintf('%f \n',size(images));
#fprintf('%f \n',size(labels));
[m,n] = size(images);
image = zeros(28,28,n);
fprintf('start loading data ...\n');
for i = 1 : n,
  #fprintf('%f \n',i);
  image(:,:,i) = reshape(images(:,i),28,28);
end;
fprintf('data load done \n');
clear images;
#===========================================================================================================================
filter_1 = 0.01 .* randn(5,5,20);
bias_1 = 0.01 .* randn(20,1);
conv_out_1 = zeros(24,24,20);
max_index_1 = zeros(24,24,20);
max_pool_1 = zeros(12,12,20);
#===========================================================================================================================
filter_2 = 0.01 .* randn(3,3,20,40);
bias_2 = 0.01 .* randn(40,1);
conv_out_2 = zeros(10,10,40);
max_index_2 = zeros(10,10,40);
max_pool_2 = zeros(5,5,40);
#===========================================================================================================================
filter_3 = 0.01 .* randn(3,3,40,60);
bias_3 = 0.01 .* randn(60,1);
conv_out_3 = zeros(3,3,60);
#===========================================================================================================================
w1 = 0.01 .* randn(160,540);
b1 = 0.01 .* rand(160,1);
w2 = rand(10,160);
b2 = rand(10,1);
dbais_3 = zeros(60,1);
dfilter_3 = zeros(3,3,40,60);
dmax_pool_2 = zeros(5,5,40);

vw1 = zeros(160,540);
vb1 = zeros(160,1);
vw2 = zeros(10,160);
vb2 = zeros(10,1);
vfilter_3 = zeros(3,3,40,60);
vbais_3 = zeros(60,1);
vfilter_2 = zeros(3,3,20,40);
vbais_2 = zeros(40,1);
vfilter_1 = zeros(5,5,20);
vbais_1 = zeros(20,1);
# 0.1625   85%
# 0.1053   90%
# 0.05129  95%
# 0.01     99%
min_loss = 5;
iterate = 0;

t=[0];
m=[min_loss];
plot = plot(t,m,'EraseMode','background','MarkerSize',5);
axis([0 50 -2.5 2.5]);
grid on;

while min_loss > 0.01
#fprintf('start conv \n');
for i = 1:20,
  conv_out_1(:,:,i) = max(0,convn(image(:,:,1),filter_1(:,:,i),'valid') .+ bias_1(i,1));
  [max_pool_1(:,:,i),max_index_1(:,:,i)] = maxPooling(conv_out_1(:,:,i));
end;
#fprintf('conv1 done \n');
for i = 1:40,
  conv_out_2(:,:,i) = max(0,convn(max_pool_1,filter_2(:,:,:,i),'valid') .+ bias_2(i,1));
  [max_pool_2(:,:,i),max_index_2(:,:,i)] = maxPooling(conv_out_2(:,:,i));
end;
#fprintf('conv2 done \n');
for i = 1:60,
  conv_out_3(:,:,i) = max(0,convn(max_pool_2,filter_3(:,:,:,i),'valid') .+ bias_3(i,1));
end;
#fprintf('conv3 done \n');
#===========================================================================================================================
x = conv_out_3(:);

z1 = w1*x + b1;
a1 = max(0,z1);
#fprintf('full done \n');
#===========================================================================================================================

z2 = w2*a1 + b2;
a2 = z2;
p = softmax(a2);
#fprintf('softmax done \n');
#===========================================================================================================================
loss = -log(p(labels(1)));
min_loss = loss;
iterate ++;
fprintf('iterate = %d loss = %f\n',iterate,loss)

t=[t iterate];
m=[m loss];
set(plot,'XData',t,'YData',m);
pause(0.01);

#===========================================================================================================================
db2 = p;
db2(labels(1)) = db2(labels(1)) - 1;
dw2 = db2 * a1';

da1 = w2' * db2;
dz1 = grelu(a1) .* da1;
db1 = dz1;
dw1 = db1 * x';
#dx = w1' * db1;
#===========================================================================================================================

#rfilter_3 = rot90(rot90(filter_3));

#gconv3 = grelu3(conv_out_3);
#dconv_out_3 = reshape(dx,3,3,60) .* gconv3;
#for i = 1 : 60,
#  dbais_3(i,1) = sum(sum(dconv_out_3(:,:,i)));
#  for j = 1 : 40,
#    dfilter_3(:,:,j,i) = dfilter_3(:,:,j,i) + convn(max_pool_2(:,:,j),dconv_out_3(:,:,i),'valid');
#    dmax_pool_2(:,:,j) = dmax_pool_2(:,:,j) + convn(dconv_out_3(:,:,i),rfilter_3(:,:,j,i),'full');
#  end;
#end;
#===========================================================================================================================
vw1 = 0.9 .* vw1 + 0.1 .* dw1;
vb1 = 0.9 .* vb1 + 0.1 .* db1;
vw2 = 0.9 .* vw2 + 0.1 .* dw2;
vb2 = 0.9 .* vb2 + 0.1 .* db2;
#vfilter_3 = 0.9 * vfilter_3 + 0.1 * dfilter_3;
#vbais_3 = 0.9 * vbais_3 + 0.1 * dbais_3;
#===========================================================================================================================
w1 = w1 - 0.01 * vw1;
w2 = w2 - 0.01 * vw2;
b1 = b1 - 0.01 * vb1;
b2 = b2 - 0.01 * vb2;
#filter_3 = filter_3 - 0.1 * vfilter_3;
#bias_3 = bias_3 - 0.1 * vbais_3;
end;
fprintf("end\n");
#kron
#a = [1 2;3 4];
#b = [1 1;1 1];
#c = kron(a,b);
end;







