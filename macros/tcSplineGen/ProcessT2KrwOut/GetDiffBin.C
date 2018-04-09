int GetDiffBin(float P, float Cos, int nPBins, int nCosBins, float* PBins, float** CosBins){

  int OutBin=-1;
  int PBin=-1;
  int CosBin=-1;

  //int nPBins=11;
  //int nCosBins=7;


  //float PBins[11]={0,200,400,550,690,830,1070,1490,2250,3000,5000};
  //nPBins=11 nCosBins=7 nBins=77
  /*
  float PBins[12]={0,200,400,550,690,830,1070,1490,2250,3000,4000,5000};
  float CosBins[11][8]={
    {-1.0,0.6,0.8,0.8,0.84,0.896,0.952,1},
    {-1.0,0.6,0.8,0.8,0.84,0.896,0.952,1},
    {-1.0,0.5,0.72,0.78,0.868,0.912,0.956,1},
    {-1.0,0.5,0.66,0.84,0.892,0.932,0.968,1},
    {-1.0,0.5,0.68,0.86,0.908,0.94,0.972,1},
    {-1.0,0.6,0.74,0.876,0.92,0.948,0.976,1},
    {-1.0,0.7,0.8,0.9,0.936,0.96,0.98,1},
    {-1.0,0.8,0.85,0.936,0.96,0.976,0.988,1},
    {-1.0,0.85,0.90,0.96,0.976,0.984,0.992,1},
    {-1.0,0.85,0.90,0.96,0.976,0.984,0.992,1},
    {-1.0,0.85,0.90,0.96,0.976,0.984,0.992,1},
  };
*/

  if(P<0) return -1;
  if(Cos<-2) return -1;



  //for(int i=0; i<10; i++){
  for(int i=0; i<nPBins; i++){
    if(P>=PBins[i] && P<PBins[i+1]){
      PBin=(i+1);
      break;
    }
  }

  if(PBin<1){
    //cout<<"Failed to find P Bin"<<endl;
    return -1;
  }

  //for(int i=0; i<6; i++){
  for(int i=0; i<nCosBins; i++){
    if(Cos>=CosBins[PBin-1][i] && Cos<CosBins[PBin-1][i+1]){
      CosBin=(i+1);
      break;
    }
  }

  if(CosBin<1){
    cout<<"Failed to find Cos Bin"<<endl;
    cout << Cos << " " << P << endl;
    return -1;
  }

  //OutBin=(PBin-1)*6+(CosBin-1);
  OutBin=(PBin-1)*(nCosBins)+(CosBin-1);

  if(OutBin<0){
    cout << "something went horribly wrong" << endl;
    return -1;
  }
  else return OutBin;

}
