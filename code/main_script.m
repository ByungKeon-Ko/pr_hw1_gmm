% sample.mat : N1 = 1000, N2 = 1500
% each data is 4-dim, and with label, one column consists of 5 elements
% Class 1 and 2 consist of 3 GMM, 2GMM each.
%    : GMM1~2 --> class1, GMM3~5 --> class2

load '../data/sample.mat'
load '../data/testData.mat'

nTRAINSET = 2500;
nTESTSET = 1000;
N1 = 1000;
N2 = 1500;
nGMM = 5;
lenX = 4;
fid = fopen('../output/nll_graph.txt', 'w');

%% ---------------------------------------------------------------- %%
%% STEP1 : Initialization
%% ---------------------------------------------------------------- %%
mu1 = zeros(4,1);
mu2 = zeros(4,1);
mu3 = zeros(4,1);
mu4 = zeros(4,1);
mu5 = zeros(4,1);

% allset = sample( randperm(nTRAINSET), :);
% --------------- Problem : some GMMs link to wrong classes.
% --------------- Soln : It can be solved by initial grouping.
allset = sample;
class1_set = sample( randperm(N1), :);
class2_set = sample( randperm(N2)+N1, :);

gmm_set1 = class1_set(   1: 500, : );
gmm_set2 = class1_set( 501:1000, : );
gmm_set3 = class2_set(   1: 500, : );
gmm_set4 = class2_set( 501:1000, : );
gmm_set5 = class2_set(1001:1500, : );

mu1 = mean( gmm_set1(:,1:4) );
mu2 = mean( gmm_set2(:,1:4) );
mu3 = mean( gmm_set3(:,1:4) );
mu4 = mean( gmm_set4(:,1:4) );
mu5 = mean( gmm_set5(:,1:4) );

sigma1 = cov(allset(:, 1:4));
sigma2 = sigma1;
sigma3 = sigma1;
sigma4 = sigma1;
sigma5 = sigma1;
mu = [mu1; mu2; mu3; mu4; mu5];
sigma = zeros( nGMM, lenX, lenX);
sigma(1,:,:) = sigma1;
sigma(2,:,:) = sigma2;
sigma(3,:,:) = sigma3;
sigma(4,:,:) = sigma4;
sigma(5,:,:) = sigma5;

% P( theta ) : Priori
for i =1:nGMM
	phi(i) = 500/2500.;
end

%% ---------------------------------------------------------------- %%
%% STEP2 : Training
%% ---------------------------------------------------------------- %%
W = zeros(nTRAINSET, nGMM);
nll = 1e7;

for iter = 1:1e3
	%% Expectation
	% P( x | theta ) : likelihood
	pdf = zeros(nTRAINSET, nGMM);

	sigma_tmp = zeros( lenX, lenX);
	for j = 1:nGMM
		sigma_tmp(:,:) = sigma(j,:,: );
		pdf(:,j) = MultGaussDist( allset(:,1:4), mu(j,:), sigma_tmp );
	end

	% P( x | theta ) * P(theta) : similar to Posterior
	posterior = bsxfun(@times, pdf, phi);
	W = bsxfun(@rdivide, posterior, sum(posterior, 2) );

	%% Maximization
	for j = 1:nGMM

		phi(j) = mean(W(:,j),1);
		mu(j, :) = weightedAverage( W(:,j), allset(:,1:4) );

		Xm = bsxfun(@minus, allset(:,1:4), mu(j,:) );

		sigma_tmp = zeros( lenX, lenX);
		for i = 1:nTRAINSET
			sigma_tmp = sigma_tmp + ( W(i,j) .* (Xm(i,:)' *Xm(i,:)));
		end
		sigma(j,:,:) = sigma_tmp ./ sum( W(:,j) );
	end

	pre_nll = nll;
	% nll = -sum( sum(log(W(:,j)), 1) );
	nll = -sum( sum(log(pdf(:,j)), 1) );

	% Monitoring
	if mod(iter, 2) == 0
		expect_class = MakeDecision( sample, mu, sigma, phi );
		acc = ComputeAcc( expect_class, sample );
    	fprintf('  EM Iteration %d, ', iter);
    	fprintf(' NLL = %f, train acc = %f \n', nll, acc);
    	fprintf(fid, '%d %f %f\n', iter, nll, acc);
	end

	if (pre_nll <= nll ) & (iter >50 )
    	fprintf('Converge Done! \n');
		break
	end
end

%% ---------------------------------------------------------------- %%
%% STEP3 : Test
%% ---------------------------------------------------------------- %%
expect_class = MakeDecision( testData, mu, sigma, phi );

output_data = [testData, expect_class'];
save('../output/testData_2013_20000.mat', 'output_data');

%% ---------------------------------------------------------------- %%
%% STEP4 : NLL Graph
%% ---------------------------------------------------------------- %%
load '../output/nll_graph.txt';
figure(1);
hold on;
plot(nll_graph(:,1), nll_graph(:,2) );
title('NLL graph');
xlabel('iter');
ylabel('NLL');
hold off;


