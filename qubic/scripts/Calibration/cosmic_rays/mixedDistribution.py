import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import rv_continuous, norm, t, cauchy, uniform, kstest

plt.rcParams['text.usetex'] = True


class MixedDistribution:
    """
    The class fits a two-component mixed distribution to the residuals computed as (model - data) / data. It calculates key statistical measures based on the mixed distribution fit. The class accepts distributions derived from scipy.stats.rv_continuous (see https://docs.scipy.org/doc/scipy/reference/stats.html). Because the residuals are typically expected to follow a Gaussian distribution, one of the two components is often chosen as Gaussian. However, to maintain the generality of the class, both component distributions can be selected arbitrarily.

    Args
    ---------
    dist1           (rv_continuous): first distribution which constitutes the mixed-distribution fit
    dist2           (rv_continuous): second distribution which constitutes the mixed-distribution fit

    Attributes
    -----------
    (All of the above)

    name        (str): distribution name, given by the concatenation of the dist1 and dist2 names
    params      (tuple): distribution parameters (dist1 + dist2)
    len_params  (list): length of parameters (dist1 + dist2)
    nll:        (callable): negative log-likelihood function is defined as the negation of the logarithm of the probability of                       reproducing a given data set, which is used in the Maximum Likelihood method to determine model
                            parameters
    """

    def __init__(self,
                 dist1: rv_continuous,
                 dist2: rv_continuous):

        # attributes
        self.dist1 = dist1
        self.dist2 = dist2

        self.name: str = dist1.name.capitalize() + dist2.name.capitalize()
        self.params: tuple = ()
        self.len_params: list = []
        self.nll: callable = None

    def mean(self, *args):
        """
        Calculates the mean of the mixed distribution obtained from the fit.
        The function determines the necessary parameters to compute the mean,
        either using those already stored in self.params or by accepting them as arguments.

        Parameters
        ----------
        *args
            If self.params is not defined, the following parameters must be provided:
            - The first argument is the weight (alpha) for the first distribution
            - The subsequent arguments are the parameters for the first distribution
            - The remaining arguments are the parameters for the second distribution

        Returns
        -------
        float
            The mean of the mixed distribution, calculated as the weighted sum of the means
            of the two component distributions
        """

        # if parameters are already stored in self.params, extract them
        if self.params:
            # extract alpha (weight for the first distribution), params1, and params2
            alpha, params1, params2 = self.params

        # otherwise, if arguments are provided, use them to define the parameters
        elif args:
            # the first argument is the weight alpha for the first distribution
            alpha = args[0]
            # the following arguments are the parameters for the first distribution
            params1 = args[1: self.len_params[0] + 1]
            # the remaining arguments are the parameters for the second distribution
            params2 = args[self.len_params[0] + 1:]

        # if no parameters are available in self.params or through args, raise an exception
        else:
            raise RuntimeError("Perform the fit or provide the parameters")

        # calculate the mean of the first distribution using the parameters in params1
        dist1_mean = self.dist1(*params1).mean()
        # calculate the mean of the second distribution using the parameters in params2
        dist2_mean = self.dist2(*params2).mean()

        # return the mean of the mixed distribution as a weighted combination
        return alpha * dist1_mean + (1 - alpha) * dist2_mean

    def std(self, *args):
        """
        Calculates the standard deviation of the mixed distribution.

        The function retrieves the parameters either from self.params or from the provided arguments,
        computes the means and standard deviations of the two component distributions, and then calculates
        the overall standard deviation of the mixed distribution.

        Parameters
        ----------
        *args : tuple
            If self.params is not already set, the parameters must be provided as follows:
            - The first argument is the weight (alpha) for the first distribution
            - The next set of arguments are the parameters for the first distribution
            - The remaining arguments are the parameters for the second distribution

        Returns
        -------
        float
            The standard deviation of the mixed distribution
        """

        # if parameters are available in self.params, extract them
        if self.params:
            alpha, params1, params2 = self.params

        # otherwise, if arguments are provided, extract parameters from args
        elif args:
            # the first argument is the weight (alpha) for the first distribution
            alpha = args[0]
            # the next arguments correspond to the parameters for the first distribution
            params1 = args[1: self.len_params[0] + 1]
            # the remaining arguments correspond to the parameters for the second distribution
            params2 = args[self.len_params[0] + 1:]

        # if no parameters are available in self.params or through args, raise an exception
        else:
            raise RuntimeError("Perform the fit or provide the parameters")

        # calculate the mean of the first distribution using params1
        dist1_mean = self.dist1(*params1).mean()
        # calculate the mean of the second distribution using params2
        dist2_mean = self.dist2(*params2).mean()

        # calculate the standard deviation of the first distribution using params1
        dist1_sigma = self.dist1(*params1).std()
        # calculate the standard deviation of the second distribution using params2
        dist2_sigma = self.dist2(*params2).std()

        # if either standard deviation is not finite, return NaN
        if not (np.isfinite(dist1_sigma) and np.isfinite(dist2_sigma)):
            return np.nan

        # calculate the second moment of each component distribution.
        moment_dist1 = dist1_sigma ** 2 + dist1_mean ** 2
        moment_dist2 = dist2_sigma ** 2 + dist2_mean ** 2

        # calculate the second moment of the mixed distribution as the weighted
        # sum of the components' second moments
        mixed_moment = alpha * moment_dist1 + (1 - alpha) * moment_dist2

        # Compute the variance of the mixed distribution.
        # Note: Typically, variance = E[X^2] - (E[X])^2.
        # Here it is computed as the difference between the mixed second moment
        # and the squared mixed mean.
        variance = mixed_moment - self.mean() ** 2

        # guarantee a non-negative variance
        variance = 0 if variance < 0 else variance

        # return the standard deviation as the square root of the variance
        return np.sqrt(variance)

    def pdf(self, data, *args):
        """
        Computes the probability density function (PDF) of the mixed distribution.

        This method is used both during the fitting process and after the fit, when the parameters
        have been obtained and can be passed to the PDF.

        Parameters
        ----------
        data : array-like
            The input data for which the PDF is evaluated
        *args : tuple
            If self.params is not set, parameters must be provided as follows:
            - The first argument is the weight (alpha) for the first distribution
            - The next arguments are the parameters for the first distribution
            - The remaining arguments are the parameters for the second distribution

        Returns
        -------
        array-like
            The PDF of the mixed distribution, computed as a weighted sum of the PDFs of the two components
        """

        # if the parameters are already stored in self.params, extract them
        if self.params:
            alpha, params1, params2 = self.params

        # otherwise, if parameters are provided as arguments, extract them accordingly
        elif args:
            # the first argument is the weight (alpha) for the first distribution
            alpha = args[0]
            # the subsequent arguments are the parameters for the first distribution
            params1 = args[1: self.len_params[0] + 1]
            # the remaining arguments are the parameters for the second distribution
            params2 = args[self.len_params[0] + 1:]

        # if no parameters are available in self.params or through args, raise an exception
        else:
            raise RuntimeError("Perform the fit or provide the parameters")

        # compute the PDF of the first distribution using the provided data and parameters
        pdf_dist1 = self.dist1.pdf(data, *params1)
        # compute the PDF of the second distribution using the provided data and parameters
        pdf_dist2 = self.dist2.pdf(data, *params2)

        # return the weighted sum of the two PDFs
        return alpha * pdf_dist1 + (1 - alpha) * pdf_dist2

    def cdf(self, data, *args):
        """
        Computes the cumulative distribution function (CDF) of the mixed distribution.

        This method is used both during the fitting process and after the fit, when the parameters
        have been determined and can be passed to the CDF.

        Parameters
        ----------
        data : array-like
            The input data at which the CDF is evaluated
        *args : tuple
            If self.params is not set, the following parameters must be provided:
            - The first argument is the weight (alpha) for the first distribution
            - The subsequent arguments are the parameters for the first distribution
            - The remaining arguments are the parameters for the second distribution

        Returns
        -------
        array-like
            The CDF of the mixed distribution, computed as a weighted sum of the CDFs of the two components
        """

        # if the parameters are already stored in self.params, extract them
        if self.params:
            alpha, params1, params2 = self.params

        # otherwise, if parameters are provided as arguments, extract them accordingly
        elif args:
            # the first argument is the weight (alpha) for the first distribution
            alpha = args[0]
            # the subsequent arguments are the parameters for the first distribution
            params1 = args[1: self.len_params[0] + 1]
            # the remaining arguments are the parameters for the second distribution
            params2 = args[self.len_params[0] + 1:]

        # if no parameters are available in self.params or through args, raise an exception
        else:
            raise RuntimeError("Perform the fit or provide the parameters")

        # compute the CDF for the first distribution using the provided data and parameters
        pdf_dist1 = self.dist1.cdf(data, *params1)
        # compute the CDF for the second distribution using the provided data and parameters
        pdf_dist2 = self.dist2.cdf(data, *params2)

        # return the weighted sum of the two CDFs
        return alpha * pdf_dist1 + (1 - alpha) * pdf_dist2

    def kstest(self, data):
        """
        Performs the Kolmogorov-Smirnov (KS) test on the provided data against the mixed distribution.

        This function uses the cumulative distribution function (CDF) of the mixed distribution
        to evaluate how well the provided data fits the model. It prints the KS statistic and p-value,
        and returns the test result.

        Parameters
        ----------
        data : array-like
            The sample data to test against the mixed distribution

        Returns
        -------
        ks_result : object
            The result of the KS test, containing attributes such as 'statistic' and 'pvalue'
        """

        # perform the KS test using the mixed distribution's CDF
        ks_result = kstest(data, self.cdf)
        print(f"KS ~ statistic: {ks_result.statistic:.2f}")
        print(f"KS ~ p-value: {ks_result.pvalue:.2e}")

        # return the full KS test result
        return ks_result

    def compute_AIC_BIC(self, data):
        """
        Computes the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC)
        for the mixed distribution model.
        AIC and BIC are statistical measures used for model selection.
        They balance the goodness of fit of the model with its complexity (i.e., the number of parameters).
        Lower values indicate a better model that achieves a good fit without being overly complex.

        The function uses the negative log-likelihood (nll) stored in the instance to calculate the log-likelihood.
        Then, it computes AIC and BIC based on the total number of parameters and the number of data points.

        Parameters
        ----------
        data : array-like
            The dataset used for model fitting, from which the number of data points is derived.

        Returns
        -------
        tuple
            A tuple (AIC, BIC) where:
            - AIC is the Akaike Information Criterion.
            - BIC is the Bayesian Information Criterion.
        """

        # compute the log-likelihood by taking the negative of the negative log-likelihood
        logL = - self.nll

        # calculate AIC using the formula: AIC = 2*k - 2*logL,
        # where k is the total number of parameters (sum of parameters across distributions).
        AIC = 2 * np.sum(self.len_params) - 2 * logL

        # calculate BIC using the formula: BIC = k*log(n) - 2*logL,
        # where n is the number of data points in the dataset.
        BIC = np.sum(self.len_params) * np.log(len(data)) - 2 * logL

        # return the computed AIC and BIC
        return AIC, BIC

    def plot_pp(self, data):
        """
        Generates a Probability-Probability (P-P) plot to compare the theoretical CDF
        In this plot, the theoretical CDF is computed by evaluating the model's CDF on the sorted data,
        which ties it directly to the observed data values. The empirical CDF, on the other hand, is
        derived from the data ranks. Comparing these two helps assess how well the model fits the data.

        Parameters
        ----------
        data : array-like
            The observed data used to compute the empirical CDF
        """

        # sort the data in ascending order
        data_sorted = sorted(data)
        n = len(data)

        # compute the empirical CDF: each data point's rank divided by the total number of points
        emp_cdf = np.arange(1, n + 1) / n
        # compute the theoretical CDF by evaluating the model on the sorted data
        th_cdf = self.cdf(data_sorted)

        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
        ax.plot(th_cdf, emp_cdf, 'o', markersize=3, label=f'${self.name}$')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set(xlabel=r'$Theoretical ~ CDF$', ylabel=r'$Empirical ~ CDF$', title=f'$Probability-Probability ~ Plot$')
        ax.legend(loc='best')
        ax.grid(True)

    def fit(self, data, **kwargs):
        """
        Fits the mixed distribution model to the provided data by minimizing the negative log-likelihood.

        This function estimates the parameters of the mixed distribution model using numerical optimization.
        It first initializes parameter estimates for each component distribution separately, then combines them
        with a mixing weight (alpha) and minimizes the negative log-likelihood of the model with respect to the data.

        Parameters
        ----------
        data : array-like
            The observed data used to fit the model
        **kwargs : dict
            Optional keyword arguments to customize the fitting process:
            - p0: Initial guess for the parameters
            - bounds: Bounds for the parameters
            - method: Optimization method (default is "COBYQA")

        Returns
        -------
        tuple
            A tuple containing the fitted parameters: (alpha, params1, params2), where:
            - alpha is the weight for the first distribution
            - params1 and params2 are the parameters for the first and second distributions, respectively
        """

        # Estimate initial parameters separately for each distribution.
        # Here, use the mean of the data for each parameter as a simple initial guess.
        params1_initial = [np.mean(data)] * len(self.dist1.fit(data))
        params2_initial = [np.mean(data)] * len(self.dist2.fit(data))

        # Get the number of parameters for each distribution
        p1_len = len(params1_initial)
        p2_len = len(params2_initial)

        # store the lengths for later use in parameter extraction
        self.len_params = [p1_len, p2_len]

        # Retrieve initial parameter guess, bounds, and optimization method from kwargs, if provided.
        # p0 defaults to [0.5, *params1_initial, *params2_initial], where 0.5 is the initial guess for alpha
        p0 = kwargs.get("p0", [0.5, *params1_initial, *params2_initial])
        # Bounds default: alpha between 0 and 1, and no bounds for the other parameters
        bounds = kwargs.get("bounds", [(0, 1)] + [(None, None)] * (p1_len + p2_len))
        # Default optimization method is set to "COBYQA"
        method = kwargs.get("method", "COBYQA")

        # Set options based on the chosen optimization method
        if method == "Powell":
            options = {'disp': False, 'maxfev': 20000}  # da usare con Powell
        elif method == "TNC":
            options = {'disp': False, 'maxfun': 20000}  # da usare con TNC
        else:
            options = {'disp': False, 'maxiter': 10000}  # da usare con gli altri

        def neg_log_likelihood(params):
            """
            Computes the negative log-likelihood for the mixed distribution model.

            Parameters
            ----------
            params : array-like
                The parameter vector containing alpha, parameters for the first distribution (p1),
                and parameters for the second distribution (p2).

            Returns
            -------
            float
                The negative log-likelihood value. Returns infinity if parameters are invalid

            Notes
            -----
            This function is defined as an inner function within the fit method to leverage variables from the outer
            scope, such as 'p1_len', 'p2_len', and 'data'. This approach simplifies the code by avoiding the need to pass
            these variables explicitly as parameters, and it restricts the scope of neg_log_likelihood to the fit method,
            as it is only relevant there.
            """

            # extract the mixing weight alpha and the parameters for each distribution
            alpha = params[0]
            p1 = params[1:p1_len + 1]
            p2 = params[p1_len + 1:]

            # compute the mixed PDF for the data using the current parameters
            mixed_pdf = self.pdf(data, alpha, *p1, *p2)

            # if alpha is not between 0 and 1 or any PDF value is non-positive, return infinity
            if not (0 <= alpha <= 1) or np.any(mixed_pdf <= 0):
                return np.inf

            # return the negative log-likelihood: the negative sum of the log of the mixed PDF
            return -np.sum(np.log(mixed_pdf))

        # perform the optimization to minimize the negative log-likelihood
        result = minimize(neg_log_likelihood, p0, bounds=bounds, method=method, options=options)

        # if the optimization did not succeed, raise an error with the failure message
        if not result.success:
            # raise RuntimeError(f"Fit does not converge: {result.message}")
            print(f"Optimization failed with message: {result.message}")
            return None


        # store the negative log-likelihood of the fitted model
        self.nll = result.fun

        # extract the optimal parameters from `result`
        # `result.x` contains the fitted parameters in the same order as they were passed
        alpha_opt = result.x[0]
        params1_opt = result.x[1:p1_len + 1]
        params2_opt = result.x[p1_len + 1:]

        # save the fitted parameter
        self.params = (alpha_opt, params1_opt, params2_opt)

        return self.params