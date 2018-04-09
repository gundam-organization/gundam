int makeModelCovMat(){

  // *********************************
  // *** Det syst ********************
  // Manipulate HL2 matrix to only include the cov mat elements that we care about
  // *********************************

  // Coarse binning:

  string fdetcov_fine = inputDir + "/detCovMatFine.root";
  string fdetcov = inputDir + "/detCovMatCoarse.root";
  TFile *findetcov = TFile::Open(fdetcov.c_str()); //contains det. systematics info
  TFile *findetcov_fine = TFile::Open(fdetcov_fine.c_str()); //contains det. systematics info
  const int ndetcovmatele = 24;
  const int ndetcovmatele_fine = 48;

  double arr[ndetcovmatele];
  for(int i=0; i<ndetcovmatele; i++){ arr[i]=(1.0);}

  TVectorD* det_weights = new TVectorD(ndetcovmatele, arr);
  det_weights->Print();

  TMatrixDSym *cov_det_now   = (TMatrixDSym*)findetcov->Get("covMat_norm");
  TMatrixDSym *cov_det_in   = new TMatrixDSym(24);
  cov_det_now->GetSub(0,(ndetcovmatele-1),*cov_det_in);
  cov_det_in->Print();

  if(!cov_det_in) cout << "Warning! Problem opening detector cov matrix" << endl;
  TMatrixDSym cov_det(cov_det_in->GetNrows());
  for(size_t m=0; m<cov_det_in->GetNrows(); m++){
    for(size_t k=0; k<cov_det_in->GetNrows(); k++){
      cov_det(m, k) = (*cov_det_in)(m,k);
      //May need to add small terms to the last sample to make thh matrix invertable:
      //if((m>19)&&(k>19)&&(m==k)) cov_det(m, k) = 1.1*((*cov_det_in)(m,k));
    }
  }

  cov_det.SetTol(1e-50);
  cov_det.Print();
  double det = cov_det.Determinant();

  if(abs(det) < 1e-50){
    cout << "Warning, det cov matrix is non invertable. Det is:" << endl;
    cout << det << endl;
    return 0;   
  }  


  findetcov->Close();

  // ***** Fine binning: ******


  double arr_fine[ndetcovmatele_fine];
  for(int i=0; i<ndetcovmatele_fine; i++){ arr_fine[i]=(1.0);}
  TVectorD* det_weights_fine = new TVectorD(ndetcovmatele_fine, arr_fine);
  det_weights_fine->Print();

  TMatrixDSym *cov_det_now_fine   = (TMatrixDSym*)findetcov_fine->Get("covMat_norm");
  TMatrixDSym *cov_det_in_fine   = new TMatrixDSym(ndetcovmatele_fine);
  cov_det_now_fine->GetSub(0,(ndetcovmatele_fine-1),*cov_det_in_fine);
  cov_det_in_fine->Print();

  TMatrixDSym cov_det_fine(cov_det_in_fine->GetNrows());
  for(size_t m=0; m<cov_det_in_fine->GetNrows(); m++){
    for(size_t k=0; k<cov_det_in_fine->GetNrows(); k++){
      cov_det_fine(m, k) = (*cov_det_in_fine)(m,k);
    }
  }

  cov_det_fine.SetTol(1e-50);
  cov_det_fine.Print();
  double det_fine = cov_det_fine.Determinant();

  if(abs(det_fine) < 1e-50){
    cout << "Warning, det cov matrix with fine binning is non invertable. Det is:" << endl;
    cout << det_fine << endl;
    return 0;   
  }  

  findetcov_fine->Close();

  // *********************************
  // *** XSEC syst *******************
  // *********************************


  TMatrixDSym cov_xsec(9);

  // Cov mat for xsec

  cov_xsec(0,0) = 0.01412; //CA5Res
  cov_xsec(0,1) = 0.0;
  cov_xsec(0,2) = 0.0;
  cov_xsec(0,3) = 0.0;
  cov_xsec(0,4) = 0.0;
  cov_xsec(0,5) = 0.0;
  cov_xsec(0,6) = 0.0;
  cov_xsec(0,7) = 0.0;
  cov_xsec(0,8) = 0.0;

  cov_xsec(1,1) = 0.02493; //MARES
  cov_xsec(1,2) = 0;
  cov_xsec(1,3) = 0;
  cov_xsec(1,4) = 0;
  cov_xsec(1,5) = 0;
  cov_xsec(1,6) = 0;
  cov_xsec(1,7) = 0;
  cov_xsec(1,8) = 0;

  cov_xsec(2,2) = 0.02367; //BgRES
  cov_xsec(2,3) = 0;
  cov_xsec(2,4) = 0;
  cov_xsec(2,5) = 0;
  cov_xsec(2,6) = 0;
  cov_xsec(2,7) = 0;
  cov_xsec(2,8) = 0;

  cov_xsec(3,3) = 0.0004; //CCNUE_0
  cov_xsec(3,4) = 0;
  cov_xsec(3,5) = 0;
  cov_xsec(3,6) = 0;
  cov_xsec(3,7) = 0; 
  cov_xsec(3,8) = 0;

  cov_xsec(4,4) = 0.16; //dismpishp
  cov_xsec(4,5) = 0;
  cov_xsec(4,6) = 0;
  cov_xsec(4,7) = 0;
  cov_xsec(4,8) = 0;

  cov_xsec(5,5) = 1.0; //CCCOH
  cov_xsec(5,6) = 0;
  cov_xsec(5,7) = 0;
  cov_xsec(5,8) = 0;

  cov_xsec(6,6) = 0.09; //NCCOH
  cov_xsec(6,7) = 0;
  cov_xsec(6,8) = 0;
  
  cov_xsec(7,7) = 0.09; //NCOTHER
  cov_xsec(7,8) = 0;

  cov_xsec(8,8) = 0.1296; //Eb_C12 

  cov_xsec.SetTol(1e-50);
  cov_xsec.Print();
  double det_cov_xsec = cov_xsec.Determinant();

  if(abs(det_cov_xsec) < 1e-50){
    cout << "Warning, xsec cov matrix is non invertable. Det is:" << endl;
    cout << det_cov_xsec << endl;
    return 0;   
  }  

  // *********************************
  // *** FSI syst ********************
  // *********************************

  // Taken from BANFF prefit matrix
  TMatrixDSym cov_fsi(6);
  cov_fsi(0,0)= 0.17;
  cov_fsi(0,1)= -0.002778;
  cov_fsi(0,2)= 0;
  cov_fsi(0,3)= 0.02273;
  cov_fsi(0,4)= 0.005;
  cov_fsi(0,5)= 0;
  cov_fsi(1,1)= 0.1142;
  cov_fsi(1,2)= -0.1667;
  cov_fsi(1,3)= -0.001263;
  cov_fsi(1,4)= -0.002083;
  cov_fsi(1,5)=  -0.09259;
  cov_fsi(2,2)= 0.25;
  cov_fsi(2,3)= -5.204e-18;
  cov_fsi(2,4)= 0;
  cov_fsi(2,5)= 0.1389;
  cov_fsi(3,3)= 0.1694;
  cov_fsi(3,4)= -0.002273;
  cov_fsi(3,5)= -3.469e-18;
  cov_fsi(4,4)= 0.3213;
  cov_fsi(4,5)= 1.735e-18;
  cov_fsi(5,5)= 0.07716;

  // //Need to add 0.001 to the diagonal, otherwise not positive definite
   cov_fsi(0,0)= cov_fsi(0,0)+0.001;
   cov_fsi(1,1)= cov_fsi(1,1)+0.001;
   cov_fsi(2,2)= cov_fsi(2,2)+0.001;
   cov_fsi(3,3)= cov_fsi(3,3)+0.001;
   cov_fsi(4,4)= cov_fsi(4,4)+0.001;
   cov_fsi(5,5)= cov_fsi(5,5)+0.001;

   cov_fsi.SetTol(1e-50);
   cov_fsi.Print();
   double det_cov_fsi = cov_fsi.Determinant();

   if(abs(det_cov_fsi) < 1e-50){
     cout << "Warning, pion FSI cov matrix is non invertable. Det is:" << endl;
     cout << det_cov_fsi << endl;
     return 0;   
   }  

   // *************
   // Write Output
   // *************

   TFile * outputf = new TFile("makeModelCovMatOut.root", "RECREATE");
   cov_det->Write("cov_det");
   cov_det_fine->Write("cov_det_fine");
   cov_xsec->Write("cov_xsec");
   cov_fsi->Write("cov_fsi");
   outputf->Close();

}
