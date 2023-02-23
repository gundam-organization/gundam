//
// Created by Adrien Blanchet on 22/02/2023.
//

#ifndef GUNDAM_PARAMETERTHROWERMARKHARZ_H
#define GUNDAM_PARAMETERTHROWERMARKHARZ_H

#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "TF1.h"


class ParameterThrowerMarkHarz {

public:
  ParameterThrowerMarkHarz(TVectorD &parms, TMatrixDSym &covm);
  ~ParameterThrowerMarkHarz();

  void ThrowSet(std::vector<double> &parms);
  static void StdNormRand(double *z);
  void CheloskyDecomp(TMatrixD &chel_mat);

private:
  int npars{-1};
  TVectorD* pvals{nullptr};
  TMatrixDSym* covar{nullptr};
  TMatrixD* chel_dec{nullptr};
  TF1* gauss{nullptr};

};


#endif //GUNDAM_PARAMETERTHROWERMARKHARZ_H
