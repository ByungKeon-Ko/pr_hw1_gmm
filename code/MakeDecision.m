%% Calculate Accuracy

function expect_class = MakeDecision( test_set, mu, sigma, phi)

nGMM = 5;
lenX = 4;

nDATA = size(test_set, 1);

% find expected class ( decision )
W_test = zeros(nDATA, nGMM);
pdf = zeros(nDATA, nGMM);

sigma_tmp = zeros( lenX, lenX);
for j = 1:nGMM
	sigma_tmp(:,:) = sigma(j,:,: );
	pdf(:,j) = MultGaussDist( test_set(:,1:4), mu(j,:), sigma_tmp );
end

% P( x | theta ) * P(theta) : similar to Posterior
% posterior = bsxfun(@times, pdf, phi);
% W_test = bsxfun(@rdivide, posterior, sum(posterior, 2) );
W_test = bsxfun(@rdivide, pdf, sum(pdf, 2) );

[expect_prob expect_index] = max( W_test, [], 2);

gmm_map = [ 1, 1, 2, 2, 2];
expect_class = gmm_map(expect_index);

end

