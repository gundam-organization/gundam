#ifndef CALCCHISQ_HH
#define CALCCHISQ_HH

#include <iostream>
#include <vector>

#include "TDecompLU.h"
#include "TH1D.h"
#include "TMath.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"

#include "ColorOutput.hh"

class CalcChisq
{
    public:
        CalcChisq();
        CalcChisq(const TMatrixD& cov);
        CalcChisq(const TMatrixDSym& cov);
        ~CalcChisq();

        void SetCovariance(const TMatrixD& cov);
        void SetCovariance(const TMatrixDSym& cov);

        double CalcChisqCov(const TH1D& h1, const TH1D& h2) const;
        double CalcChisqStat(const TH1D& h1, const TH1D& h2) const;

    private:
        bool did_invert;
        double tol;
        unsigned int npar;
        TMatrixD* cov_mat;
        TMatrixD* inv_mat;

        const std::string TAG = color::MAGENTA_STR + "[CalcChisq]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + color::BOLD_STR
                                + "[ERROR]: " + color::RESET_STR;
};

#endif
