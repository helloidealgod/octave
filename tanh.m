figure('NumberTitle', 'off', 'Name', 'Tanh函数');
x=-5:0.1:5;
y=2./(1+exp(-2*x))-1;
plot(x,y);
xlabel('X轴');ylabel('Y轴');%坐标轴表示对象标签
grid on;%显示网格线
axis on;%显示坐标轴
axis([-5,5,-1,1]);%x,y的范围限制
title('Tanh函数');