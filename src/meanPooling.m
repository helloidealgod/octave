function mean = meanPooling(matrix)
  [m,n] = size(matrix);
  mean = zeros(m/2,n/2);
  for i = 1 : m/2,
    for j = 1 : n/2,
      temp =  matrix((i-1)*2+1,(j-1)*2+1)+ matrix((i-1)*2+1,(j-1)*2+2)+ matrix((i-1)*2+2,(j-1)*2+1)+ matrix((i-1)*2+2,(j-1)*2+2);
      mean(i,j) = temp / 4;
    end;
  end;
end;