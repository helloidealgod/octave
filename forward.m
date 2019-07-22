#前向计算
#   sample(28,28,1)
#   filter1(5,5,1,20),bias(20,1)
#   filter2(3,3,20,40),bias2(40,1)
#   filter3(3,3,40,60),bias3(60,1)
#   w1(160,540),b1(160,1)
#   w2(10,160),b2(10,1)
#
function [c1,m1,i1,c2,m2,i2,c3,z1,a1,z2,p,loss] = forward(sample,filter1,filter2,filter3,w1,w2,bias1,bias2,bias3,b1,b2,label)

#计算卷积层1
for i = 1:20,
  c1(:,:,i) = max(0,convn(sample(:,:,1),filter1(:,:,:,i),'valid') .+ bias1(i,1));
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
loss = -log(p(label));
return;
end;