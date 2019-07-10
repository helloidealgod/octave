function image = main()
images = loadMNISTImages("t10k-images.idx3-ubyte");
labels = loadMNISTLabels("t10k-labels.idx1-ubyte");
#fprintf('%f \n',size(images));
#fprintf('%f \n',size(labels));
[m,n] = size(images);
image = zeros(28,28,n);
for i = 1 : n,
  fprintf('%f \n',i);
  image(:,:,i) = reshape(images(:,i),28,28);
end;
# c1 = zeros(24,24,3);
# c1(:,:,3) = convn(image,k1(:,:,3),'valid');
end;