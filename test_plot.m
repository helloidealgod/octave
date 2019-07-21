t=[0]
m=sin(t)
p = plot(t,m,'EraseMode','background','MarkerSize',5);
x=-1.5*pi;
#axis( [xmin xmax ymin ymax] )    设置当前坐标轴 x轴 和 y轴的限制范围
axis([x x+2*pi -1.5 1.5]);
#打开网格
grid on;

for i=1:1000
    t=[t 0.1*i];  
    m=[m sin(0.1*i)]; 
    set(p,'XData',t,'YData',m)   
    drawnow
    x=x+0.1;    
    #axis([0 x+2*pi -1.5 1.5]);
    axis([x x+2*pi -1.5 1.5]);
    pause(0.01);
end