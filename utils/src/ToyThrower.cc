#include "ToyThrower.hh"

ToyThrower::ToyThrower(int nrows, unsigned int seed)
    : npar(nrows), force_limit(100)
{
    covmat = new TMatrixD(npar, npar);
    L_matrix = new TMatrixD(npar, npar);
    R_vector = new TVectorD(npar);
    RNG = new TRandom3(seed);
}

ToyThrower::ToyThrower(const TMatrixD &cov, unsigned int seed, double decomp_tol)
    : ToyThrower(cov.GetNrows(), seed)
{
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

    SetupDecomp(decomp_tol);
}

ToyThrower::ToyThrower(const TMatrixDSym &cov, unsigned int seed, double decomp_tol)
    : ToyThrower(cov.GetNrows(), seed)
{
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*covmat)[i][j] = cov[i][j];
        }
    }

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
