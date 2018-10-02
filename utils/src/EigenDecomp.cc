#include "EigenDecomp.hh"

EigenDecomp::EigenDecomp(const TVectorD& nom, const TMatrixD& cov, Method flag = kEigen)
    : npar(cov.GetNrows())
    , decomp_method(flag)
    , eigen_vectors(nullptr)
    , eigen_vectorsI(nullptr)
    , eigen_covmat(nullptr)
    , eigen_nominal(nullptr)
    , eigen_values(nullptr)
{
    SetupDecomp(nom, cov);
}

EigenDecomp::EigenDecomp(const TVectorD& nom, const TMatrixDSym& cov, Method flag = kEigen)
    : npar(cov.GetNrows())
    , decomp_method(flag)
    , eigen_vectors(nullptr)
    , eigen_vectorsI(nullptr)
    , eigen_covmat(nullptr)
    , eigen_nominal(nullptr)
    , eigen_values(nullptr)
{
    SetupDecomp(nom, cov);
}

EigenDecomp::EigenDecomp(const std::vector<double>& nom, const TMatrixD& cov, Method flag = kEigen)
    : EigenDecomp(TVectorD(nom.size(), &nom[0]), cov, flag)
{
}

EigenDecomp::EigenDecomp(const std::vector<double>& nom, const TMatrixDSym& cov, Method flag = kEigen)
    : EigenDecomp(TVectorD(nom.size(), &nom[0]), cov, flag)
{
}

EigenDecomp::~EigenDecomp()
{
    if(eigen_vectors != nullptr)
        delete eigen_vectors;

    if(eigen_vectorsI != nullptr)
        delete eigen_vectorsI;

    if(eigen_nominal != nullptr)
        delete eigen_nominal;

    if(eigen_values != nullptr)
        delete eigen_values;

    if(eigen_covmat != nullptr)
        delete eigen_covmat;
}

template<typename Matrix>
void EigenDecomp::SetupDecomp(const TVectorD& nom, const Matrix& cov)
{
    eigen_nominal = new TVectorD(npar);
    eigen_covmat  = new TMatrixD(npar, npar);

    if(decomp_method == kEigen)
    {
        eigen_values   = new TVectorD(0);
        eigen_vectors  = new TMatrixD(cov.EigenVectors(*eigen_values));
        eigen_vectorsI = new TMatrixD(cov.EigenVectors(*eigen_values));
        eigen_vectorsI->Invert();
    }
    else if(decomp_method == kSVD)
    {
        TDecompSVD svd(cov);
        eigen_values   = new TVectorD(svd.GetSig());
        eigen_vectors  = new TMatrixD(svd.GetU());
        eigen_vectorsI = new TMatrixD(svd.GetU());
        eigen_vectorsI->T();
    }
    else
    {
        std::cout << "[ERROR]: Invalid method in EigenDecomp()." << std::endl;
    }

#ifdef DEBUG_MSG
    std::cout << "ev: " << eigen_values->GetNrows() << std::endl;
    std::cout << "en: " << eigen_nominal->GetNrows() << std::endl;
    std::cout << "eV: " << eigen_vectors->GetNrows() << std::endl;
    std::cout << "eI: " << eigen_vectorsI->GetNrows() << std::endl;
    std::cout << "eM: " << eigen_covmat->GetNrows() << std::endl;
    std::cout << "nm: " << nom.GetNrows() << std::endl;
    std::cout << "np: " << npar << std::endl;
#endif

    (*eigen_nominal) = (*eigen_vectorsI) * nom;

    eigen_covmat->Zero();
    for(int i = 0; i < npar; ++i)
    {
        if((*eigen_values)(i) > 0.0)
            (*eigen_covmat)(i, i) = (*eigen_values)(i);
    }
}

std::vector<double> EigenDecomp::GetEigenNominalSTL() const
{
    auto arr = eigen_nominal->GetMatrixArray();
    return std::vector<double>(arr, arr + npar);
}

std::vector<double> EigenDecomp::GetEigenValuesSTL() const
{
    auto arr = eigen_values->GetMatrixArray();
    return std::vector<double>(arr, arr + npar);
}

TVectorD EigenDecomp::GetOriginalParameters(const TVectorD& param) const
{
    return (*eigen_vectors) * param;
}

TVectorD EigenDecomp::GetDecompParameters(const TVectorD& param) const
{
    return (*eigen_vectorsI) * param;
}

std::vector<double> EigenDecomp::GetOriginalParameters(const std::vector<double>& param) const
{
    std::vector<double> result(npar, 0.0);
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            result[i] += (*eigen_vectors)(i, j) * param[j];
        }
    }

    return result;
}

std::vector<double> EigenDecomp::GetDecompParameters(const std::vector<double>& param) const
{
    std::vector<double> result(npar, 0.0);
    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            result[i] += (*eigen_vectorsI)(i, j) * param[j];
        }
    }

    return result;
}
