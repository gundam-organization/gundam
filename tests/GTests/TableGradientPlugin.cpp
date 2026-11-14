// Small external-library fixture for the Tabulated/Kriged factory ABI.
extern "C" int initializeGradientTable(const char*, int, const char**, int){ return 4; }
extern "C" int updateGradientTable(const char*, double* values_, int bins_, const double* par_, int npar_){
  if( npar_ != 2 ){ return 1; }
  for( int i = 0; i < bins_; ++i ){ values_[i] = (i + 1.) * par_[0] * par_[0] + (i - 1.) * par_[1]; }
  return 0;
}
extern "C" int differentiateGradientTable(const char*, double* derivatives_, int bins_, const double* par_, int npar_){
  if( npar_ != 2 ){ return 1; }
  for( int i = 0; i < bins_; ++i ){
    derivatives_[i * npar_] = 2. * (i + 1.) * par_[0];
    derivatives_[i * npar_ + 1] = i - 1.;
  }
  return 0;
}
extern "C" double binGradientTable(const char*, int, double*, int){ return 1.25; }
extern "C" int weightGradientTable(const char*, int, int, double*, int maxEntries_, int* index_, double* weights_){
  if( maxEntries_ < 3 ){ return -1; }
  index_[0] = 0; index_[1] = 2; index_[2] = 3;
  weights_[0] = .5; weights_[1] = -.25; weights_[2] = .75;
  return 3;
}
