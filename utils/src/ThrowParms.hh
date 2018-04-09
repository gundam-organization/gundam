//////////////////////////////////////////////////////////
//
//  Class for drawing random vectors from a multi-
//  normal distribution. This is essentially
//  BANFF implementation by Mark Hartz
//
//  Created:
//  Modified:
//
//////////////////////////////////////////////////////////
#ifndef __THROWPARAM_HH__
#define __THROWPARAM_HH__

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <math.h>

#include <TRandom3.h>
#include <TMath.h>
#include <TMatrixT.h>
#include <TMatrixTSym.h>
#include <TVectorT.h>
#include <TDecompChol.h>

#include <vector>
#include <iostream>

class ThrowParms 
{
 private:
  typedef TVectorT<double> TVectorD;
  typedef TMatrixTSym<double> TMatrixDSym;
  typedef TMatrixT<double> TMatrixD;

  TVectorD    *pvals;
  TMatrixDSym *covar; 
  TMatrixD    *chel_dec;
  int         npars;

public:
  ThrowParms(TVectorD &parms, TMatrixDSym &covm);
  ~ThrowParms();
  int GetSize() {return npars;};
  void ThrowSet(std::vector<double> &parms);

private:
  void CheloskyDecomp(TMatrixD &chel_mat);
  void StdNormRand(double *z);
};
#endif
