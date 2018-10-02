#include "AnaFitParameters.hh"

AnaFitParameters::AnaFitParameters()
    : m_name("")
    , Npar(0)
    , m_rng_priors(false)
    , covariance(nullptr)
    , covarianceI(nullptr)
{
}

AnaFitParameters::~AnaFitParameters()
{
    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;
}

void AnaFitParameters::SetCovarianceMatrix(const TMatrixDSym& covmat)
{
    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;

    covariance  = new TMatrixDSym(covmat);
    covarianceI = new TMatrixDSym(covmat);
    covarianceI->SetTol(1e-200);

    double det = 0;
    covarianceI->Invert(&det);
    /*
    if(abs(det) < 1e-200)
    {
        std::cerr << "[ERROR]: In AnaFitParameters::SetCovarianceMatrix():\n"
                  << "[ERROR]: Covariance matrix is non invertable. Determinant is " << det
                  << std::endl;
        return;
    }
    */
    std::cout << "[SetCovarianceMatrix]: Covariance matrix size: " << covariance->GetNrows()
              << " x " << covariance->GetNrows() << " for " << this->m_name << std::endl;
    /*
    std::cout << "[SetCovarianceMatrix]: Inverted Cov mat: " << std::endl;
    covarianceI->Print();
    std::cout << "[SetCovarianceMatrix]: Cov mat: " << std::endl;
    covariance->Print();
    */
}

double AnaFitParameters::GetChi2(const std::vector<double>& params)
{
    if(covariance == nullptr)
        return 0.0;

    if(CheckDims(params) == false)
    {
        std::cout << "[WARNING]: In AnaFitParameters::GetChi2()\n"
                  << "[WARNING]: Dimension check failed. Returning zero." << std::endl;
        return 0.0;
    }

    double chi2 = 0;
    for(int i = 0; i < covarianceI->GetNrows(); i++)
    {
        for(int j = 0; j < covarianceI->GetNrows(); j++)
        {
            chi2
                += (params[i] - pars_prior[i]) * (params[j] - pars_prior[j]) * (*covarianceI)(i, j);
        }
    }

    return chi2;
}

bool AnaFitParameters::CheckDims(const std::vector<double>& params)
{
    bool vector_size = false;
    bool matrix_size = false;

    if(params.size() == pars_prior.size())
    {
        vector_size = true;
    }

    else
    {
        std::cerr << "[ERROR]: Dimension of parameter vector does not match priors.\n"
                  << "[ERROR]: Prams size is: " << params.size() << std::endl
                  << "[ERROR]: Prior size is: " << pars_prior.size() << std::endl;
        vector_size = false;
    }

    if(covariance->GetNrows() == pars_prior.size())
    {
        matrix_size = true;
    }

    else
    {
        std::cerr << "[ERROR]: Dimension of covariance maxtix does not match priors.\n"
                  << "[ERROR]: Rows in cov mat: " << covariance->GetNrows() << std::endl
                  << "[ERROR]: Prior size is: " << pars_prior.size() << std::endl;
        matrix_size = false;
    }

    return vector_size && matrix_size;
}
