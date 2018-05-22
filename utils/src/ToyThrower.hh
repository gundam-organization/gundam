#ifndef TOYTHROWER_HH
#define TOYTHROWER_HH

#include <iostream>

#include "TDecompChol.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TRandom3.h"
#include "TVectorT.h"

typedef TMatrixT<double> TMatrixD;
typedef TMatrixTSym<double> TMatrixDSym;
typedef TVectorT<double> TVectorD;

class ToyThrower
{
    private:
        TMatrixD* covmat;
        TMatrixD* L_matrix;
        TVectorD* R_vector;
        TRandom3* RNG;

        unsigned int npar;

        ToyThrower(int nrows, unsigned int seed = 0);

    public:
        ToyThrower(const TMatrixD &cov, unsigned int seed = 0, double decomp_tol = 0xCAFEBABE);
        ToyThrower(const TMatrixDSym &cov, unsigned int seed = 0, double decomp_tol = 0xCAFEBABE);
        ~ToyThrower();

        void SetSeed(unsigned int seed);
        void SetupDecomp(double decomp_tol = 0xCAFEBABE);

        void Throw(TVectorD& toy);
        void Throw(std::vector<double>& toy);
};

#endif
