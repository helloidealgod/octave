function [c1,p1,position1,c2,p2,position2,c3,f2,fs,out] = main()
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
k1 = randn(5,5,20);
k2 = randn(3,3,40);
k3 = randn(3,3,60);
c1 = zeros(24,24,20);
c2 = zeros(10,10,40);
c3 = zeros(3,3,60);
position1 = zeros(24,24,20);
p1 = zeros(12,12,20);
position2 = zeros(10,10,40);
p2 = zeros(5,5,40);
fprintf('start conv \n');
for i = 1:20,
  c1(:,:,i) = convn(image(:,:,1),k1(:,:,i),'valid');
  [p1(:,:,i),position1(:,:,i)] = maxPooling(c1(:,:,i));
end;
fprintf('conv1 done \n');
for i = 1:40,
  c2(:,:,i) = convn(p1(:,:,1),k2(:,:,i),'valid');
  [p2(:,:,i),position2(:,:,i)] = maxPooling(c2(:,:,i));
end;
fprintf('conv2 done \n');
for i = 1:60,
  c3(:,:,i) = convn(p2(:,:,1),k3(:,:,i),'valid');
end;
fprintf('conv3 done \n');
f1 = c3(:);
w1 = randn(160,540);
b1 = rand(160,1);
f2 = w1*f1 + b1;
fprintf('full done \n');
ws = rand(10,160);
bs = rand(10,1);
fs = ws*f2 + bs;
out = softmax(fs);
fprintf('softmax done \n');
end;