#include "ToyThrower.hh"

ToyThrower::ToyThrower(int nrows, unsigned int seed)
    : npar(nrows), force_limit(100)
{
    covmat = new TMatrixD(npar, npar);
    L_matrix = new TMatrixD(npar, npar);
    R_vector = new TVectorD(npar);
    RNG = new TRandom3(seed);
}

ToyThrower::ToyThrower(const TMatrixD &cov, unsigned int seed, bool do_setup, double decomp_tol)
    : ToyThrower(cov.GetNrows(), seed)
{
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

    if(do_setup)
        SetupDecomp(decomp_tol);
}

ToyThrower::ToyThrower(const TMatrixDSym &cov, unsigned int seed, bool do_setup, double decomp_tol)
    : ToyThrower(cov.GetNrows(), seed)
{
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

    if(do_setup)
        SetupDecomp(decomp_tol);
}

ToyThrower::~ToyThrower()
{
    delete covmat;
    delete L_matrix;
    delete R_vector;
    delete RNG;
}

void ToyThrower::SetSeed(unsigned int seed)
{
    delete RNG;
    RNG = new TRandom3(seed);
}

void ToyThrower::SetMatrix(const TMatrixD& cov)
{
    std::cout << TAG << "Setting matrix." << std::endl
              << TAG << "Npar = " << cov.GetNrows() << std::endl;

    npar = cov.GetNrows();
    covmat->ResizeTo(npar, npar);

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

    L_matrix->ResizeTo(npar, npar);
    L_matrix->Zero();
}

void ToyThrower::SetMatrix(const TMatrixDSym& cov)
{
    std::cout << TAG << "Setting matrix." << std::endl
              << TAG << "Npar = " << cov.GetNrows() << std::endl;

    npar = cov.GetNrows();
    covmat->ResizeTo(npar, npar);

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

    L_matrix->ResizeTo(npar, npar);
    L_matrix->Zero();

    R_vector->ResizeTo(npar);
}


bool ToyThrower::SetupDecomp(double decomp_tol)
{
    TDecompChol decomp(*covmat);
    if(decomp_tol != 0xCAFEBABE)
    {
        decomp.SetTol(decomp_tol);
        std::cout << TAG << "Setting tolerance: "
                  << decomp_tol << std::endl;
    }

    if(!decomp.Decompose())
    {
        std::cerr << ERR << "Failed to decompose uncertainty matrix."
                  << std::endl;
        return false;
    }

    (*L_matrix) = decomp.GetU();
    (*L_matrix) = L_matrix -> Transpose(*L_matrix);

    std::cout << TAG << "Decomposition successful." << std::endl;
    return true;
}

bool ToyThrower::ForcePosDef(double val, double decomp_tol)
{
    std::cout << TAG << "Forcing positive-definite..." << std::endl;
    double total_add = 0;
    unsigned int limit = 0;
    while(!SetupDecomp(decomp_tol))
    {
        if(limit++ > force_limit)
        {
            std::cout << ERR << "Reached iteration limit of " << force_limit
                      << std::endl;
            return false;
        }

        for(int i = 0; i < covmat->GetNrows(); ++i)
            (*covmat)(i,i) += val;
        total_add += val;
    }
    std::cout << TAG << "Added " << total_add << " to force positive-definite."
              << std::endl;
    return true;
}

void ToyThrower::Throw(TVectorD& toy)
{
    if(toy.GetNrows() != npar)
    {
        std::cerr << ERR << "Toy vector is not the same size as covariance matrix."
                  << ERR << "Vector: " << toy.GetNrows() << " vs Matrix: " << npar
                  << std::endl;
        return;
    }

    for(int i = 0; i < npar; ++i)
        (*R_vector)[i] = RNG -> Gaus();

    toy.Zero();
    for(int j = 0; j < npar; ++j)
    {
        for(int k = 0; k < npar; ++k)
        {
            toy[j] += (*L_matrix)[j][k] * (*R_vector)[k];
        }
    }
}

void ToyThrower::Throw(std::vector<double>& toy)
{
    if(toy.size() != npar)
    {
        std::cerr << ERR << "Toy vector is not the same size as covariance matrix.\n"
                  << ERR << "Vector: " << toy.size() << " vs Matrix: " << npar
                  << std::endl;
        return;
    }

    for(int i = 0; i < npar; ++i)
        (*R_vector)[i] = RNG -> Gaus();

    std::fill(toy.begin(), toy.end(), 0);
    for(int j = 0; j < npar; ++j)
    {
        for(int k = 0; k < npar; ++k)
        {
            toy[j] += (*L_matrix)[j][k] * (*R_vector)[k];
        }
    }
}

double ToyThrower::ThrowSinglePar(double nom, double err) const
{
    return RNG -> Gaus(nom, err);
}

bool ToyThrower::CholDecomp()
{
    TMatrixD chol_mat(*covmat);

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            double sum = chol_mat(i,j);
            for(int k = 0; k < i; ++k)
                sum -= chol_mat(i,k)*chol_mat(j,k);

            if(i == j)
            {
                if(sum <= 0.0)
                {
                    std::cout << ERR << "Matrix not positive definite. Decomposition failed." << std::endl;
                    return false;
                }

                chol_mat(i,j) = std::sqrt(sum);
            }

            else
                chol_mat(j,i) = sum / chol_mat(i,i);
        }
    }

    for(int i = 0; i < npar; ++i)
        for(int j = i+1; j < npar; ++j)
            chol_mat(i,j) = 0.0;

    std::cout << TAG << "Decomposition successful." << std::endl;

    (*L_matrix) = chol_mat;
    return true;
}

bool ToyThrower::IncompCholDecomp(const double tol, bool modify_cov)
{
    TMatrixD chol_mat(*covmat);

    if(modify_cov)
    {
        std::cout << TAG << "Modifying covariance matrix for decomposition." << std::endl
                  << TAG << "Setting elements with abs(val) less than " << tol << " to zero."
                  << std::endl;

        for(int i = 0; i < npar; ++i)
        {
            for(int j = 0; j < npar; ++j)
            {
                if(std::fabs(chol_mat(i,j)) < tol)
                    chol_mat(i,j) = 0.0;
            }
        }
    }

    std::cout << TAG << "Performing incomplete Cholesky decomposition." << std::endl
              << TAG << "Dropout tolerance is " << tol << std::endl;

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            double sum = chol_mat(i,j);
            for(int k = 0; k < i; ++k)
                sum -= chol_mat(i,k)*chol_mat(j,k);

            if(i == j)
            {
                if(sum <= 0.0)
                {
                    std::cout << ERR << "Matrix not positive definite. Decomposition failed." << std::endl;
                    return false;
                }

                chol_mat(i,j) = std::sqrt(sum);
            }

            else
            {
                if(std::fabs(chol_mat(j,i)) > tol)
                    chol_mat(j,i) = sum / chol_mat(i,i);
                else
                    chol_mat(j,i) = 0.0;
            }
        }
    }

    for(int i = 0; i < npar; ++i)
        for(int j = i+1; j < npar; ++j)
            chol_mat(i,j) = 0.0;

    std::cout << TAG << "Decomposition successful." << std::endl;

    (*L_matrix) = chol_mat;
    return true;
}
