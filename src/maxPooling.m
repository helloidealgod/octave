function [max,position] = maxPooling(matrix)
  [m,n] = size(matrix);
  position = zeros(m,n);
  max = zeros(m/2,n/2);
  for i = 1 : m/2,
    for j = 1 : n/2,
      ind1 = 1;
      if matrix((i-1)*2+1,(j-1)*2+1) < matrix((i-1)*2+1,(j-1)*2+2),
        ind1 = 2;
      end;
      ind2 = 1;
      if matrix((i-1)*2+2,(j-1)*2+1) < matrix((i-1)*2+2,(j-1)*2+2),
        ind2 = 2;
      end;
      if matrix((i-1)*2+1,(j-1)*2+ind1) >= matrix((i-1)*2+2,(j-1)*2+ind2),
        position((i-1)*2+1,(j-1)*2+ind1) = 1;
        max(i,j) = matrix((i-1)*2+1,(j-1)*2+ind1);
      else
        position((i-1)*2+2,(j-1)*2+ind2) = 1;
        max(i,j) = matrix((i-1)*2+2,(j-1)*2+ind2);
      end;
    end;
  end;
end;