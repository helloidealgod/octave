function out = softmax(in)
sum = sum(exp(in));
out = in./sum;
end;