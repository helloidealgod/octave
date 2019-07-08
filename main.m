function main()
images = loadMNISTImages("t10k-images.idx3-ubyte");
labels = loadMNISTLabels("t10k-labels.idx1-ubyte");
fprintf('%f \n',size(images));
fprintf('%f \n',size(labels));
end;