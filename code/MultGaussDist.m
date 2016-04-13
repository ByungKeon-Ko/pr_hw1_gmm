function [ pdf ] = MultGaussDist(X, mu, Sigma)

lenX = 4;

% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
pdf = 1 / sqrt((2*pi)^4 * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));

end

