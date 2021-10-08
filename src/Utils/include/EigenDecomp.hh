#ifndef EIGENDECOMP_HH
#define EIGENDECOMP_HH

#include <iostream>
#include <vector>

#include "TDecompSVD.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVectorT.h"

using TMatrixD    = TMatrixT<double>;
using TMatrixDSym = TMatrixTSym<double>;
using TVectorD    = TVectorT<double>;

enum Method
{
    kEigen,
    kSVD
};

class EigenDecomp
{
private:
    unsigned int npar;
    Method decomp_method;
    TMatrixD* eigen_vectors;
    TMatrixD* eigen_vectorsI;
    TVectorD* eigen_values;
    TMatrixDSym* eigen_covmat;

    template<typename Matrix>
    void SetupDecomp(const Matrix& cov);

public:
    EigenDecomp(const TMatrixD& cov, Method flag = kEigen);
    EigenDecomp(const TMatrixDSym& cov, Method flag = kEigen);
    ~EigenDecomp();

    TVectorD GetEigenValues() const { return *eigen_values; }
    TMatrixD GetEigenVectors() const { return *eigen_vectors; }
    TMatrixDSym GetEigenCovMat() const { return *eigen_covmat; }

    double GetEigenVectors(const int i, const int j) const { return (*eigen_vectors)(i, j); }
    double GetEigenValues(const int i) const { return (*eigen_values)(i); }

    std::vector<double> GetEigenValuesSTL() const;
    int GetInfoFraction(const double frac) const;

    TVectorD GetOriginalParameters(const TVectorD& param) const;
    TVectorD GetDecompParameters(const TVectorD& param) const;
    const std::vector<double> GetOriginalParameters(const std::vector<double>& param) const;
    const std::vector<double> GetOriginalParameters(const std::vector<double>& param,
                                                    unsigned int start_idx) const;
    const std::vector<double> GetDecompParameters(const std::vector<double>& param) const;
    const std::vector<double> GetDecompParameters(const std::vector<double>& param,
                                                  unsigned int start_idx) const;

    template<typename Matrix>
    TMatrixDSym GetOriginalCovMat(const Matrix& cov_decomp, unsigned int start_idx) const
    {
        const unsigned int cov_size = cov_decomp.GetNrows();
        const unsigned int end_idx  = start_idx + npar;

        if(end_idx > cov_size)
        {
            std::cout << "[ERROR]: In EigenDecomp::GetOriginalCovMat()"
                      << "is greater than cov matrix size." << std::endl
                      << "[ERROR]: Returning original matrix." << std::endl;
            return cov_decomp;
        }

        TMatrixD V(cov_size, cov_size);
        for(unsigned int i = 0; i < cov_size; ++i)
        {
            for(unsigned int j = 0; j < cov_size; ++j)
            {
                if(i >= start_idx && i < end_idx && j >= start_idx && j < end_idx)
                    V(i, j) = (*eigen_vectors)(i - start_idx, j - start_idx);
                else
                    V(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
        TMatrixD VI(V);
        VI.Invert();

        TMatrixD temp = V * cov_decomp * VI;

        return TMatrixDSym(temp.GetNrows(), temp.GetMatrixArray());
    }
};

#endif
