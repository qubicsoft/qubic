Non Linear Component Map Making  by Victor Chabirand 2024-09-07

- acquisition_non_linear.py, which defines all the operators for the non-linear CMM.
- pipeline_non_linear.py, wich computes the gradient descent and plots the results.
- non_linear_pcg_diagonal_preconditioner.py, the pcg called by the pipeline. It works with a diagonal preconditioner.
- non_linear_pcg.py, works with a more general preconditioner, any non-linear operator, doesn’t have to be diagonal (not used).
- main.sh, used to call the pipeline
- Research_on_preconditioning, a Jupyter notebook were the Hessian matrix is computed and its eigenvalue distribution before and after preconditioning. 

Improving the preconditioner is very much needed. You will find all the information in the Appendix of my report. One has to find a better approximation of the diagonal of the Hessian operator. And note, as explained in the comments of pipeline_non_linear.py, that in the preconditioner I sum over the frequencies of Qubic instead of the pairs of frequencies. This works better, but it shouldn’t. There is something I don't understand.
