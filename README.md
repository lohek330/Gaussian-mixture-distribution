# Gaussian-mixture-distribution
Simulation code of learning a one-hidden-layer neural network with data of Gaussian mixture distribution

The file tensor_ini_perform.m is to provide a comparison between gradient descent with tensor initialization and random initialization. The result is shown in Figure 1. We cite the code from "Recovery Guarantees for One-hidden-layer Neural Networks", where they use Matlab packages, including tensor_toolbox_2.6 and tensorlab.

The file sample_to_dimension.m illustrates the relationship between the sample complexity and the feature dimension (Figure 2). We especially investigate how the mean and variance will affect the sample complexity in sample_to_mu.m and sample_to_sigma.m (Figure 3 (a) and (b)), respectively.

One can run err_mu.m and err_sigma.m to obtain Figure 4 (a) and (b), which shows the convergence rate with different mean and variance.

The file convergence_rate_K.m is to find out how the convergence rate will change when the number of hidden neurons changes (Figure 5), while critical_to_gt.m shows how the relative error of the learned model will change when the number of samples changes.
