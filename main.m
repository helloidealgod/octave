function main()
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
fprintf('start conv \n');
for i = 1:20,
  conv_out_1(:,:,i) = max(0,convn(image(:,:,1),filter_1(:,:,i),'valid') .+ bias_1(i,1));
  [max_pool_1(:,:,i),max_index_1(:,:,i)] = maxPooling(conv_out_1(:,:,i));
end;
fprintf('conv1 done \n');
for i = 1:40,
  conv_out_2(:,:,i) = max(0,convn(max_pool_1,filter_2(:,:,:,i),'valid') .+ bias_2(i,1));
  [max_pool_2(:,:,i),max_index_2(:,:,i)] = maxPooling(conv_out_2(:,:,i));
end;
fprintf('conv2 done \n');
for i = 1:60,
  conv_out_3(:,:,i) = max(0,convn(max_pool_2,filter_3(:,:,:,i),'valid') .+ bias_3(i,1));
end;
fprintf('conv3 done \n');
#===========================================================================================================================
x = conv_out_3(:);
w1 = 0.01 .* randn(160,540);
b1 = 0.01 .* rand(160,1);
z1 = w1*x + b1;
a1 = max(0,z1);
fprintf('full done \n');
#===========================================================================================================================
w2 = rand(10,160);
b2 = rand(10,1);
z2 = w2*a1 + b2;
a2 = z2;
p = softmax(a2);
fprintf('softmax done \n');
#===========================================================================================================================
loss = -log(p(labels(1)));
fprintf('loss = loss \n');
#===========================================================================================================================

db2 = p;
db2(labels(1)) = db2(labels(1)) - 1;
dw2 = db2 * a1';

da1 = w2' * db2;
dz1 = grelu(a1) .* da1;
db1 = dz1;
dw1 = db1 * x';
dx = w1' * db1;


dconv_out_3 = reshape(dx,3,3,60);
fprintf("end\n");
#kron
#a = [1 2;3 4];
#b = [1 1;1 1];
#c = kron(a,b);
end;







