function [c1,p1,position1,c2,p2,position2,c3,f2,fs,out,loss] = main()
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
k1 = 0.01 .* randn(5,5,20);
b1 = 0.01 .* randn(20,1);
k2 = 0.01 .* randn(3,3,40);
b2 = 0.01 .* randn(40,1);
k3 = 0.01 .* randn(3,3,60);
b3 = 0.01 .* randn(60,1);
c1 = zeros(24,24,20);
c2 = zeros(10,10,40);
c3 = zeros(3,3,60);
position1 = zeros(24,24,20);
p1 = zeros(12,12,20);
position2 = zeros(10,10,40);
p2 = zeros(5,5,40);
fprintf('start conv \n');
for i = 1:20,
  c1(:,:,i) = max(0,convn(image(:,:,1),k1(:,:,i),'valid') .+ b1(i,1));
  [p1(:,:,i),position1(:,:,i)] = maxPooling(c1(:,:,i));
end;
fprintf('conv1 done \n');
for i = 1:40,
  c2(:,:,i) = max(0,convn(p1(:,:,1),k2(:,:,i),'valid') .+ b2(i,1));
  [p2(:,:,i),position2(:,:,i)] = maxPooling(c2(:,:,i));
end;
fprintf('conv2 done \n');
for i = 1:60,
  c3(:,:,i) = max(0,convn(p2(:,:,1),k3(:,:,i),'valid') .+ b3(i,1));
end;
fprintf('conv3 done \n');
f1 = c3(:);
w1 = 0.01 .* randn(160,540);
bf = 0.01 .* rand(160,1);
f2 = max(0,w1*f1 + bf);
fprintf('full done \n');
ws = rand(10,160);
bs = rand(10,1);
fs = ws*f2 + bs;
out = softmax(fs);
fprintf('softmax done \n');
loss = -log(out(labels(1)));
fprintf('loss = loss \n');
#===========================================================================================================================
#kron
#a = [1 2;3 4];
#b = [1 1;1 1];
#c = kron(a,b);
dbs = out;
dbs(labels(1)) = dbs(labels(1)) - 1;
dws = dbs*f2';
dbf = ws' * dbs;
end;







