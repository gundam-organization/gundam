#include "ToyThrower.hh"

ToyThrower::ToyThrower(int nrows, unsigned int seed)
    : npar(nrows)
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

void ToyThrower::SetupDecomp(double decomp_tol)
{
    TDecompChol decomp(*covmat);
    if(decomp_tol != 0xCAFEBABE)
    {
        decomp.SetTol(decomp_tol);
        std::cout << "[ToyThrower]: Setting tolerance: "
                  << decomp_tol << std::endl;
    }

    if(!decomp.Decompose())
    {
        std::cerr << "[ERROR]: Failed to decompose uncertainty matrix."
                  << std::endl;
    }

    (*L_matrix) = decomp.GetU();
    (*L_matrix) = L_matrix -> Transpose(*L_matrix);

    std::cout << "[ToyThrower]: Decomposition successful." << std::endl;
}

void ToyThrower::Throw(TVectorD& toy)
{
    if(toy.GetNrows() != npar)
    {
        std::cerr << "[ERROR]: Toy vector is not the same size as covariance matrix."
                  << "[ERROR]: Vector: " << toy.GetNrows() << " vs Matrix: " << npar
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
        std::cerr << "[ERROR]: Toy vector is not the same size as covariance matrix.\n"
                  << "[ERROR]: Vector: " << toy.size() << " vs Matrix: " << npar
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
