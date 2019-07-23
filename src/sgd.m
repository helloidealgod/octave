addpath('./math');
addpath('./data');
#读取数据
[images,labels] = loadData();
#参数初始化
[filter1,bias1,vfilter1,vbias1,filter2,bias2,vfilter2,vbias2,filter3,bias3,vfilter3,vbias3,w1,b1,vw1,vb1,w2,b2,vw2,vb2] = init_params();
#前向计算
[c1,max_pool_1,max_index_1,c2,max_pool_2,max_index_2,c3,z1,a1,z2,p,loss] = forward(images,filter1,filter2,filter3,w1,w2,bias1,bias2,bias3,b1,b2,7);
fprintf('loss = %f \n',loss);