function [images,labels] = loadData()
  imageData = loadMNISTImages("../resource/t10k-images.idx3-ubyte");
  labels = loadMNISTLabels("../resource/t10k-labels.idx1-ubyte");
  [m,n] = size(imageData);
  images = zeros(28,28,n);
  for i = 1 : n,
    images(:,:,i) = reshape(imageData(:,i),28,28);
  end;
  clear imageData;
end;