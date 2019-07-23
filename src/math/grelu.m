function out = grule(in)
[m,n] = size(in);
out = zeros(m,n);
for i = 1 : m,
    for j = 1 : n,
        if in(i,j) > 0,
            out(i,j) = 1;
    end;
end;
end;