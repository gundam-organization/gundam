#include "EigenDecomp.hh"

EigenDecomp::EigenDecomp(const TMatrixD& cov, Method flag)
    : npar(cov.GetNrows())
    , decomp_method(flag)
    , eigen_vectors(nullptr)
    , eigen_vectorsI(nullptr)
    , eigen_covmat(nullptr)
    , eigen_values(nullptr)
{
    SetupDecomp(cov);
}

EigenDecomp::EigenDecomp(const TMatrixDSym& cov, Method flag)
    : npar(cov.GetNrows())
    , decomp_method(flag)
    , eigen_vectors(nullptr)
    , eigen_vectorsI(nullptr)
    , eigen_covmat(nullptr)
    , eigen_values(nullptr)
{
    SetupDecomp(cov);
}

EigenDecomp::~EigenDecomp()
{
    if(eigen_vectors != nullptr)
        delete eigen_vectors;

    if(eigen_vectorsI != nullptr)
        delete eigen_vectorsI;

    if(eigen_values != nullptr)
        delete eigen_values;

    if(eigen_covmat != nullptr)
        delete eigen_covmat;
}

template<typename Matrix>
void EigenDecomp::SetupDecomp(const Matrix& cov)
{
    eigen_covmat = new TMatrixDSym(npar);

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
    std::cout << "eV: " << eigen_vectors->GetNrows() << std::endl;
    std::cout << "eI: " << eigen_vectorsI->GetNrows() << std::endl;
    std::cout << "eM: " << eigen_covmat->GetNrows() << std::endl;
    std::cout << "np: " << npar << std::endl;
#endif

    eigen_covmat->Zero();
    for(int i = 0; i < npar; ++i)
    {
        if((*eigen_values)(i) > 0.0)
            (*eigen_covmat)(i, i) = (*eigen_values)(i);
    }
}

std::vector<double> EigenDecomp::GetEigenValuesSTL() const
{
    auto arr = eigen_values->GetMatrixArray();
    return std::vector<double>(arr, arr + npar);
}

int EigenDecomp::GetInfoFraction(const double frac) const
{
    int index = -1;
    if(frac >= 1.0)
        index = npar;
    else
    {
        double current_frac = 0.0;
        double integral = eigen_values -> Sum();
        for(int i = 0; i < npar; ++i)
        {
            current_frac += (*eigen_values)(i) / integral;
            if(current_frac >= frac)
            {
                index = i;
                break;
            }
        }
    }

    return index+1;
}

TVectorD EigenDecomp::GetOriginalParameters(const TVectorD& param) const
{
    return (*eigen_vectors) * param;
}

TVectorD EigenDecomp::GetDecompParameters(const TVectorD& param) const
{
    return (*eigen_vectorsI) * param;
}

const std::vector<double> EigenDecomp::GetOriginalParameters(const std::vector<double>& param) const
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

const std::vector<double> EigenDecomp::GetOriginalParameters(const std::vector<double>& param,
                                                             unsigned int start_idx) const
{
    const unsigned int param_size = param.size();
    const unsigned int end_idx    = start_idx + npar;

    TMatrixD V(param_size, param_size);
    for(unsigned int i = 0; i < param_size; ++i)
    {
        for(unsigned int j = 0; j < param_size; ++j)
        {
            if(i >= start_idx && i < end_idx && j >= start_idx && j < end_idx)
                V(i, j) = (*eigen_vectors)(i - start_idx, j - start_idx);
            else
                V(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    std::vector<double> result(param_size, 0.0);
    for(int i = 0; i < param_size; ++i)
    {
        for(int j = 0; j < param_size; ++j)
        {
            result[i] += V[i][j] * param[j];
        }
    }

    return result;
}

const std::vector<double> EigenDecomp::GetDecompParameters(const std::vector<double>& param) const
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

const std::vector<double> EigenDecomp::GetDecompParameters(const std::vector<double>& param,
                                                           unsigned int start_idx) const
{
    const unsigned int param_size = param.size();
    const unsigned int end_idx    = start_idx + npar;

    TMatrixD V(param_size, param_size);
    for(unsigned int i = 0; i < param_size; ++i)
    {
        for(unsigned int j = 0; j < param_size; ++j)
        {
            if(i >= start_idx && i < end_idx && j >= start_idx && j < end_idx)
                V(i, j) = (*eigen_vectors)(i - start_idx, j - start_idx);
            else
                V(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    V.Invert();

    std::vector<double> result(param_size, 0.0);
    for(int i = 0; i < param_size; ++i)
    {
        for(int j = 0; j < param_size; ++j)
        {
            result[i] += V[i][j] * param[j];
        }
    }

    return result;
}
