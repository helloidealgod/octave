function out = grelu3(in)
[m,n,z] = size(in);
out = zeros(m,n,z);
for i = 1 : m,
    for j = 1 : n,
      for k = 1 : z,
        if in(i,j,k) > 0,
            out(i,j,k) = 1;
      end;
    end;
end;
end;