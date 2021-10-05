## How to Throw Toys

A common technique for analyses and simulation is generating vectors of correlated variables. This is a critical part of how the error propagation code works to numerically propagate the fit parameter errors to cross-section errors.

A toy is jargon for a single iteration of the simulation, or a single vector of correlated variables. Throwing a toy is the process of calculating a new vector of correlated random variables, and optionally using that random vector in further calculation/processing. Commonly toy and throw are used interchangeably to mean the same thing.

Generating vectors of correlated random variables is a fairly short procedure, and relies on calculating the Cholesky decomposition of a covariance or correlation matrix. A covariance (or correlation) matrix has the special property that it is a positive semi-definite symmetric matrix, which implies that it (usually) has a Cholesky decomposition. Technically, a matrix needs to be positive definite to be Cholesky decomposed, but commonly the covariance matrix is positive definite. Once the Cholesky decomposition is calculated, it can be used to correlate vectors of random Gaussian distributed numbers.

The Cholesky decomposition factorizes a Hermitian, positive definite matrix \(A\) into the product of a lower triangular matrix and its conjugate transpose.

$$
A = LL^{*}
$$

In the case of matrices with only real values, \(L^{*} = L^{T}\), and also only contains real values. Implementations of Cholesky decomposition can differ in whether they return the lower triangular matrix, or the upper triangular matrix (which is simply \(U = L^{*}\) using the notation above). To correlate a random vector of Gaussian numbers (with mean zero and width one), simply multiply the vector by \(L\), which will generate a correlated vector with mean zero.

The procedure to generate correlated random toy throws where the parameters are changed by their error is as follows:

$$
\begin{align}
\Sigma &= LL^{T} \\
\vec{p_t} &= \vec{p_i} + (L \times \vec{r_t})
\end{align}
$$

where \(\Sigma\) is the covariance matrix, \(LL^{T}\) is the Choleksy decomposition, \(\vec{p_t}\) is the new toy throw parameter vector, \(\vec{p_i}\) is the initial parameter vector, and \(\vec{r_t}\) is a vector of random numbers distributed about a Gaussian of mean zero and width one. This generates random parameter vectors which are distributed according to the covariance matrix.

### Code Example

This is a short code example showing how to generate random toy throws using ROOT. It is not a complete code sample, just an illustration of the method.

```cpp
void GenerateToyThrow()
{
    //Get some matrix to decompose. Store in either TMatrixT or TMatrixTSym
    TMatrixT<double> A_matrix = some_matrix;
    TDecompChol cholesky_decomp(A_matrix);

    //Perform decomposition of A_matrix
    bool did_decompose = cholesky_decomp.Decompose();
    if(!did_decompose)
        std::cout << "Something went wrong!" << std::endl;
    //If it does not decompose, the rest of this math is invalid.

    //ROOT calculates the upper triangle, we need the lower triangle.
    //So get the upper matrix and transpose.
    TMatrixT<double> L_matrix(cholesky_decomp.GetU());
    L_matrix.T();

    //Construct some vectors to store the random numbers and toy throw
    const unsigned int nrows = L_matrix.GetNrows();
    TVectorT<double> R_vector(nrows);
    TVectorT<double> P_vector(nrows);
    P_vector.Zero();

    //Generate some Gaussian random numbers
    TRandom3 RNG();
    for(int i = 0; i < nrows; ++i)
        R_vector[i] = RNG.Gaus();

    //Perform matrix math. ROOT can do this for you,
    //but I wrote it out in case you use non-ROOT objects.
    for(int j = 0; j < nrows; ++j)
    {
        for(int k = 0; k < nrows; ++k)
        {
            P_vector[j] += L_matrix[j][k] * R_vector[k];
        }
    }

    //P_vector now contains correlated random numbers
    //according to the distribution in A
}
```
