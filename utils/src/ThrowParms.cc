#include <iostream>
#include <assert.h>
#include <algorithm>
#include <math.h>

#include "ThrowParms.hh"
  
ThrowParms::ThrowParms(TVectorD &parms, TMatrixDSym &covm)
{  
  npars = parms.GetNrows();
  //std::cout << "Number of parameters " << npars << std::endl;
  pvals = new TVectorD(npars);
  covar = new TMatrixDSym(npars);
  //parms.Print();
  (*pvals) = parms;
  (*covar) = covm;
  TDecompChol chdcmp(*covar);
  if(!chdcmp.Decompose())
  {
    std::cerr<<"ERROR: Cholesky decomposition failed"<<std::endl;
    exit(-1);
  }
  chel_dec = new TMatrixD(chdcmp.GetU());
  CheloskyDecomp((*chel_dec));
}
   
ThrowParms::~ThrowParms()
{
  if(pvals!=NULL)    pvals->Delete();
  if(covar!=NULL)    covar->Delete();
  if(chel_dec!=NULL) chel_dec->Delete();
}
   
void ThrowParms::ThrowSet(std::vector<double> &parms)
{
  if(!parms.empty()) parms.clear();
  parms.resize(npars);

  if(npars>1){
    int half_pars = npars/2+npars%2;
    TVectorD std_rand(npars);
    for(int j=0; j<half_pars; j++){
      double z[2];
      StdNormRand(z);
      std_rand(j) = z[0];
      if(npars%2==0 || j!=half_pars-1)
      std_rand(j+half_pars) = z[1];
    }
    TVectorD prod = (*chel_dec)*std_rand;
    for(int i=0;i<npars;i++) parms[i] = prod(i) + (*pvals)(i);
  }
  else{
    parms[0]=gRandom->Gaus(0,sqrt((*covar)(0,0))) + (*pvals)(0);
  }
}

void ThrowParms::StdNormRand(double *z)
{
  
  double u = 2.*gRandom->Rndm()-1.;
  double v = 2.*gRandom->Rndm()-1.;
  
  double s = u*u+v*v;
  
  while(s==0 || s>=1.){
    u = 2.*gRandom->Rndm()-1.;
    v = 2.*gRandom->Rndm()-1.;
    s = u*u+v*v;
  }
  
  z[0] = u*sqrt(-2.*TMath::Log(s)/s);
  z[1] = v*sqrt(-2.*TMath::Log(s)/s);
}

void ThrowParms::CheloskyDecomp(TMatrixD &chel_mat)
{
  for(int i=0; i<npars; i++)
  {
    for(int j=0; j<npars; j++)
    {
      //if diagonal element
      if(i==j)
      {
        chel_mat(i,i) = (*covar)(i,i);
        for(int k=0; k<=i-1; k++) chel_mat(i,i) = chel_mat(i,i)-pow(chel_mat(i,k),2);
        chel_mat(i,i) = sqrt(chel_mat(i,i));
        //if lower half
      } 
      else if(j<i) 
      {
        chel_mat(i,j) = (*covar)(i,j);
        for(int k=0; k<=j-1; k++) chel_mat(i,j) = chel_mat(i,j)-chel_mat(i,k)*chel_mat(j,k);
        chel_mat(i,j) = chel_mat(i,j)/chel_mat(j,j);
      } else chel_mat(i,j) = 0.;
    }
  }
}
 
