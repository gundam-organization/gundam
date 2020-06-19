#include "AnaFitParameters.hh"
#include "GenericToolbox.h"

AnaFitParameters::AnaFitParameters()
    : m_name("")
    , Npar(0)
    , m_rng_start(false)
    , m_do_throw(false)
    , m_decompose(false)
    , m_regularised(false)
    , m_do_cap_weights(false)
    , m_nb_dropped_dof(0)
    , m_weight_cap(1000)
    , m_info_frac(1.00)
    , m_regstrength(0.0)
    , eigen_decomp(nullptr)
    , covariance(nullptr)
    , covarianceI(nullptr)
    , original_cov(nullptr)
{
}

AnaFitParameters::~AnaFitParameters()
{
    if(eigen_decomp != nullptr)
        delete eigen_decomp;
    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;
    if(original_cov != nullptr)
        delete original_cov;
}

void AnaFitParameters::SetCovarianceMatrix(const TMatrixDSym& covmat, bool decompose)
{
    m_decompose = decompose;
    std::cout << TAG << "Using decompose ? " << m_decompose << std::endl;

    if(covmat.GetNrows() > 500){
        std::cout << WAR << "dof > 500 : Matrix inversion will take some time" << std::endl;
    }

    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;
    if(original_cov != nullptr)
        delete original_cov;
    if(eigen_decomp != nullptr)
        delete eigen_decomp;

    if(m_decompose)
    {
        eigen_decomp = new EigenDecomp(covmat, kSVD);
        original_cov = new TMatrixDSym(covmat);
        covariance   = new TMatrixDSym(eigen_decomp->GetEigenCovMat());
        covarianceI  = new TMatrixDSym(eigen_decomp->GetEigenCovMat());
    }
    else
    {
        original_cov = new TMatrixDSym(covmat);
        covariance  = new TMatrixDSym(covmat);
        covarianceI = new TMatrixDSym(covmat);
    }

    std::map<std::string, TMatrixD*> temp = GenericToolbox::SVD_matrix_inversion((TMatrixD*) covariance, "inverse_covariance_matrix:regularized_eigen_values");
    m_nb_dropped_dof = 0;
    while( (*temp["regularized_eigen_values"])[ temp["regularized_eigen_values"]->GetNrows()-1 - m_nb_dropped_dof ][0] == 0 ){
        m_nb_dropped_dof++;
    }
    if(m_nb_dropped_dof != 0){
        std::cout << WAR << "SVD matrix inversion has dropped " << m_nb_dropped_dof << " degree of freedom." << std::endl;
    }
    covarianceI = GenericToolbox::convert_to_symmetric_matrix(temp["inverse_covariance_matrix"]);
    for(const auto& content: temp) delete content.second;

//    double det = 0;
//    TMatrixD inv_matrix(*covariance);
//    if(TDecompLU::InvertLU(inv_matrix, 1E-48, &det))
//    {
//        covarianceI->SetMatrixArray(inv_matrix.GetMatrixArray());
//        std::cout << TAG << "Covariance matrix inverted successfully." << std::endl;
//    }
//    else
//    {
//        std::cerr << ERR << "In AnaFitParameters::SetCovarianceMatrix():\n"
//                  << ERR << "Covariance matrix is non invertable. Determinant is " << det
//                  << std::endl;
//        return;
//    }

    std::cout << TAG << "Covariance matrix size: " << covariance->GetNrows()
              << " x " << covariance->GetNrows() << " for " << this->m_name << std::endl;
    /*
    std::cout << "[SetCovarianceMatrix]: Inverted Cov mat: " << std::endl;
    covarianceI->Print();
    std::cout << "[SetCovarianceMatrix]: Cov mat: " << std::endl;
    covariance->Print();
    */
}

double AnaFitParameters::GetChi2(const std::vector<double>& params) const
{
    if(covariance == nullptr)
        return 0.0;

    if(CheckDims(params) == false)
    {
        std::cout << WAR << "In AnaFitParameters::GetChi2()\n"
                  << WAR << "Dimension check failed. Returning zero." << std::endl;
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

bool AnaFitParameters::CheckDims(const std::vector<double>& params) const
{
    bool vector_size = false;
    bool matrix_size = false;

    if(params.size() == pars_prior.size())
    {
        vector_size = true;
    }

    else
    {
        std::cerr << ERR << "Dimension of parameter vector does not match priors.\n"
                  << ERR << "Prams size is: " << params.size() << std::endl
                  << ERR << "Prior size is: " << pars_prior.size() << std::endl;
        vector_size = false;
    }

    if(covariance->GetNrows() == pars_prior.size())
    {
        matrix_size = true;
    }

    else
    {
        std::cerr << ERR << "Dimension of covariance maxtix does not match priors.\n"
                  << ERR << "Rows in cov mat: " << covariance->GetNrows() << std::endl
                  << ERR << "Prior size is: " << pars_prior.size() << std::endl;
        matrix_size = false;
    }

    return vector_size && matrix_size;
}

void AnaFitParameters::SetRegularisation(double strength, RegMethod flag)
{
    m_regularised = true;
    m_regstrength = strength;
    m_regmethod = flag;

    std::cout << "[AnaFitParameters]: Enabled regularisation for " << m_name << std::endl
              << "[AnaFitParameters]: Strength " << m_regstrength << std::endl;
    if(flag == kL1Reg)
        std::cout << "[AnaFitParameters]: Method L1" << std::endl;
    else if(flag == kL2Reg)
        std::cout << "[AnaFitParameters]: Method L2" << std::endl;
    else
    {
        std::cout << WAR << "In AnaFitParameters::SetRegularisation() "
                  << "Invalid regularisation method!" << std::endl;
        m_regularised = false;
    }
}

void AnaFitParameters::SetRegularisation(double strength, const std::string method)
{
    if(method == "kL1Reg")
        SetRegularisation(strength, kL1Reg);
    else if(method == "kL2Reg")
        SetRegularisation(strength, kL2Reg);
    else
    {
        std::cout << WAR << "In AnaFitParameters::SetRegularisation() "
                  << "Invalid regularisation method!" << std::endl;
    }
}

void AnaFitParameters::ThrowPar(std::vector<double>& param, const int seed) const
{
    if(covariance != nullptr && m_do_throw)
    {
        std::vector<double> toy(Npar, 0);
        ToyThrower toy_thrower(*original_cov, seed, true, 1E-48);
        toy_thrower.Throw(toy);

        std::transform(toy.begin(), toy.end(), pars_original.begin(), param.begin(),
                       std::plus<double>());
        for(auto& val : param)
        {
            if(val < 0.0)
                val = 0.0;
        }
    }
    else
    {
        param = pars_original;
    }
}
