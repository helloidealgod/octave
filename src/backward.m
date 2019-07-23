function [dfilter1,dfilter2,dfilter3,dw1,dw2,dbias1,dbias2,dbias3,db1,db2] = backward(sample,p,label,max_pool_1,max_index_1,filter2,max_pool_2,max_index_2,filter3,c3,w1,w2,z1,a1)
addpath('./math');
#softmax层梯度
db2 = p;
db2(label) = db2(label) - 1;
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
#卷积层1梯度
dc1 = zeros(24,24,20);
for i = 1 : 20,
    dc1(:,:,i) = kron(dx2(:,:,i),kw);
end;
dc1 = dc1 .* max_index_1;
dfilter1 = zeros(5,5,1,20);
for i = 1 : 20,
    dbias1 = sum(sum(dc1(:,:,i)));
    for j = 1 : 1,
    dfilter1(:,:,j,i) = dfilter1(:,:,j,i) .+ convn(sample(:,:,j),dc1(:,:,i),'valid');
    end;
end;