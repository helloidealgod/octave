function out = softmax(in)
sum = sum(exp(in));
out = exp(in)./sum;
end;