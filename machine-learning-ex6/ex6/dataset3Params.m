function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

return;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%


% potential values of C and sigma
C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% for-loop
pre_error_temp = 1;
for i = 1:length(C_array)
    this_C = C_array(i);
    
    for j = 1:length(sigma_array)
        this_sigma = sigma_array(j);
        
        % train SVM
        model= svmTrain(X, y, this_C, @(x1, x2) gaussianKernel(x1, x2, this_sigma));
        
        % make predictions using the trained SVM
        predictions = svmPredict(model, Xval);
        
        % compute prediction error
        pre_error = mean(double(predictions ~= yval));
        if pre_error < pre_error_temp
            pre_error_temp = pre_error;
            C = this_C;
            sigma = this_sigma;
        end
        
    end
    
end

fprintf('C = %f, sigma = %f.\n', C, sigma);
fprintf('prediction error on val data is %f.\n', pre_error_temp)


% =========================================================================

end
