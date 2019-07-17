function test()
x = randn(160,1);
w = 0.01 * randn(10,160);
b = 0.01 * randn(10,1);
vw = zeros(10,160);
vb = zeros(10,1);
y = [1 0 0 0 0 0 0 0 0 0];
z = w*x + b;
p = softmax(z);
loss = -log(p(1));
fprintf("loss = %f \n",loss);
while loss > 0.01,
db = p - y';
dw = db * x';
vw = 0.9 * vw + 0.1 * dw;
vb = 0.9 * vb + 0.1 * db;
w = w - 0.1 * vw;
b = b - 0.1 * vb;
z = w*x + b;
p = softmax(z);
loss = -log(p(1));
fprintf("loss = %f \n",loss);
end;
end;