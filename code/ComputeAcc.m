%% Calculate Accuracy

function acc = ComputeAcc( expect_class, test_set )

nDATA = size(test_set, 1);
compare_result = expect_class ==  test_set(:,5)' ;
acc = sum( compare_result ) / nDATA;

end

