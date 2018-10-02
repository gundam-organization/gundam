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
        TMatrixD* eigen_covmat;
        TVectorD* eigen_nominal;
        TVectorD* eigen_values;

        template<typename Matrix>
            void SetupDecomp(const TVectorD& nom, const Matrix& cov);

    public:
        EigenDecomp(const TVectorD& nom, const TMatrixD& cov, Method flag);
        EigenDecomp(const TVectorD& nom, const TMatrixDSym& cov, Method flag);
        EigenDecomp(const std::vector<double>& nom, const TMatrixD& cov, Method flag);
        EigenDecomp(const std::vector<double>& nom, const TMatrixDSym& cov, Method flag);
        ~EigenDecomp();

        TMatrixD GetEigenCovMat() const { return *eigen_covmat; }
        TMatrixD GetEigenVectors() const { return *eigen_vectors; }
        TVectorD GetEigenNominal() const { return *eigen_nominal; }
        TVectorD GetEigenValues() const { return *eigen_values; }

        double GetEigenVectors(const int i, const int j) const { return (*eigen_vectors)(i, j); }
        double GetEigenNominal(const int i) const { return (*eigen_nominal)(i); }
        double GetEigenValues(const int i) const { return (*eigen_values)(i); }

        std::vector<double> GetEigenNominalSTL() const;
        std::vector<double> GetEigenValuesSTL() const;

        TVectorD GetOriginalParameters(const TVectorD& param) const;
        TVectorD GetDecompParameters(const TVectorD& param) const;
        std::vector<double> GetOriginalParameters(const std::vector<double>& param) const;
        std::vector<double> GetDecompParameters(const std::vector<double>& param) const;
};

#endif
