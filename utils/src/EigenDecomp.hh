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

        TVectorD GetOriginalParameters(const TVectorD& param) const;
        TVectorD GetDecompParameters(const TVectorD& param) const;
        std::vector<double> GetOriginalParameters(const std::vector<double>& param) const;
        std::vector<double> GetDecompParameters(const std::vector<double>& param) const;
};

#endif
