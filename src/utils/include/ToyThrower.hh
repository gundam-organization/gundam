#ifndef TOYTHROWER_HH
#define TOYTHROWER_HH

#include <algorithm>
#include <iostream>

#include "TDecompChol.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TRandom3.h"
#include "TVectorT.h"

#include "ColorOutput.hh"

using TMatrixD = TMatrixT<double>;
using TMatrixDSym = TMatrixTSym<double>;
using TVectorD = TVectorT<double>;

class ToyThrower
{
    private:
        TMatrixD* covmat;
        TMatrixD* L_matrix;
        TVectorD* R_vector;
        TRandom3* RNG;

        unsigned int npar;
        unsigned int force_limit;

        ToyThrower(int nrows, unsigned int seed = 0);

    public:
        ToyThrower(const TMatrixD &cov, unsigned int seed = 0, bool do_setup = true, double decomp_tol = 0xCAFEBABE);
        ToyThrower(const TMatrixDSym &cov, unsigned int seed = 0, bool do_setup = true, double decomp_tol = 0xCAFEBABE);
        ~ToyThrower();

        void SetSeed(unsigned int seed);
        void SetMatrix(const TMatrixD& cov);
        void SetMatrix(const TMatrixDSym& cov);
        bool SetupDecomp(double decomp_tol = 0xCAFEBABE);
        bool ForcePosDef(double val, double decomp_tol = 0xCAFEBABE);

        bool CholDecomp();
        bool IncompCholDecomp(const double tol = 1.0E-3, bool modify_cov = true);

        void Throw(TVectorD& toy);
        void Throw(std::vector<double>& toy);

        double ThrowSinglePar(double nom, double err) const;

        const std::string TAG = color::MAGENTA_STR + "[ToyThrower]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + color::BOLD_STR
                                + "[ERROR]: " + color::RESET_STR;
};

#endif
