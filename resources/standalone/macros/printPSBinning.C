
void printPSBinning(){
  const int npmombins=3;
  const double pmombins[npmombins+1] = {0.0, 450.0, 1000.0, 100000.0};
  const int npthetabins=2;
  const double pthetabins[npthetabins+1] = {-1.0, 0.4, 1.0};
  const int nmumombins=2;
  const double mumombins[nmumombins+1] = {0.0, 250.0, 100000.0};
  const int nmuthetabins=2;
  const double muthetabins[nmuthetabins+1] = {-1.0, -0.6, 1.0};
  
  int globalBin = 1;
  
  for(int ii=0; ii<npmombins; ii++){
    for(int j=0; j<npthetabins; j++){
      for(int k=0; k<nmumombins; k++){
        for(int l=0; l<nmuthetabins; l++){
          cout << "Global Bin: " << globalBin << endl;
          cout << "Muon angular range: " << muthetabins[l] << " to " << muthetabins[l+1] << endl;
          cout << "Muon momentum range: " << mumombins[k] << " to " << mumombins[k+1] << endl;
          cout << "Proton angular range: " << pthetabins[j] << " to " << pthetabins[j+1] << endl;
          cout << "Proton momentum range: " << pmombins[ii] << " to " << pmombins[ii+1] << endl << endl;
          globalBin++;
        }
      }
    }
  }
}