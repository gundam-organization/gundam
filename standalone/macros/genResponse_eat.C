//#include "genResponse_test.h"

bool DEBUGMODE=false;

void genResponse_eat(string microtreeIn, string rwpath, string dialName, double nom, double err) {

  // declare variables for fit parameters, names, and values
  vector<string> param_names;
  vector<double> param_values;
  double bestfit;
  double error;
  //double sigma;


  // define inputs
  string microtree = microtreeIn;
  string rwfile_default = rwpath+"rw_default.root";
  string rwfile_truth = rwpath+"rw_default.root";
  //string rwfile_truth = rwpath+"rwfile_truth.root";
  string flux = "/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/fluxFiles/nd5_tuned11bv3.2_11anom_run2_fine.root";
  string fluxname = "enu_nd5_tuned11b_numu";

  string outname = Form(".root");
  string outputFileName = "./output/resFunc"+dialName+outname;
  cout << "Output file name is: " << outputFileName << endl; 
  
  string signal, cut, bcut; // these will be defined in the main loop

  // set up recon binning
  const int nRbins = 7;
  const double Rbins[nRbins+1] = { -1,-0.5,0,0.2,0.4,0.6,0.8,1};
  const int nRbins_mom = 17;
  const double Rbins_mom[nRbins_mom+1] = { 0,200,250,300,350,400,450,500,550,600,650,700,750,800,1000,1200,1500,2000};

  // set up true binning
  const int nTbins = 7;
  const double Tbins[nTbins+1] = { -1,-0.5,0,0.2,0.4,0.6,0.8,1};
  const int nTbins_mom = 17;
  const double Tbins_mom[nTbins_mom+1] = { 0,200,250,300,350,400,450,500,550,600,650,700,750,800,1000,1200,1500,2000};

  // Set up hist for global bin mapping:
  const int nTotalBins = nTbins*nTbins_mom;
  //TH3D* binMapHist = new TH3D("binMapHist","binMapHist",nTbins,Tbins,npmombins,pmombins,npthetabins,pthetabins);


  
  // delcare xsBinning, they will be used always the same in the main loop
  xsBinning  *binningTru = new xsBinning();
  xsBinning  *binningRec = new xsBinning();
  binningTru->addDimension("kHMTracker_CThetaStart",nTbins,Tbins);
  binningTru->addDimension("kHMTracker_MomStart",nTbins_mom,Tbins_mom);
  binningRec->addDimension("kHMTracker_CThetaStart",nRbins,Rbins);
  binningRec->addDimension("kHMTracker_MomStart",nRbins_mom,Rbins_mom);
  
  // setup on-the-fly reweighting parameter
  param_names.push_back(dialName);
  bestfit = nom;
  error = err;

  // Total number of targets
  const int nTargets = 3;            //C, Pb, Fe
  const double nprot_tgt[nTargets] = { 6, 82, 26};

  // Total number of weights
  //const int nWeights = 4;
  const int nWeights = 7;
  // reweighted reconstrcted distribution for each weight, true bin, and target
  TH1 *hreco[nWeights];



  int globalCount=0;

  if(DEBUGMODE) break;

  for(const int w = 0; w < nWeights; w++){

    // define xsInput
    xsInputdataHighland2 *inpRwFly = new xsInputdataHighland2();
    inpRwFly->SetupFromHighland(microtree,microtree,-1,"GlobalVtxTree","GlobalVtxTree","GlobalVtxTree");
    inpRwFly->SetFlux(flux, fluxname);
    //inpRwFly->SetRooTrackerVtxFiles(rwfile_default,rwfile_truth);
    inpRwFly->SetTrueBinning(binningTru);
    inpRwFly->SetReconstructedBinning(binningRec);

    signal = ("kHMTracker_TrueMomStart>0");
    inpRwFly->SetSignal(signal.c_str());

    // set cut definition, for each target
    cut = Form("((kAnyToyPassed) == 1)");

    //inpRwFly->SetCut((cut+" && "+bcut).c_str());
    inpRwFly->SetCut(cut.c_str());

    // define xsReweightParameter
    xsReweightParameter *rwPar = new xsReweightParameter(param_names);
    inpRwFly->AddReweighting(rwPar);

    // set parameter value
    param_values.clear();    
    param_values.push_back((bestfit-1)+((-3+w)*error));
    string name = rwPar->SetParameterValue(param_values);

    // reweighting with the given value
    if(!DEBUGMODE) inpRwFly->GenerateThrow(name,0,true);

    // get reconstructed distribution
    hreco[w] = inpRwFly->GetReconstructed();
    cout << "Reconstructed histo: w, b: " << w << ", " << globalCount << endl; 
    hreco[w]->Print("all");

    // delete rwPar
    delete rwPar;
  }
  // Increment the global bin counter
  globalCount++;
 
  //Write TGraph in the output file
  TFile *output = new TFile(outputFileName.c_str(),"RECREATE");   
  
  
  double MA[7];
  for(int w = 0; w < 7; w++){
      MA[w]=bestfit-error*(3-w);
  }  
  TGraph *ReWeight[nTotalBins];

  for(int bt = 0; bt < nTbins; bt++){//true kinematics bin
    for(int bt_mom = 0; bt_mom < nTbins_mom; bt_mom++){//true kinematics bin
      cout << "On true bin (glob, cth, p): " << (bt*nTbins_mom)+bt_mom << " " << bt << " " << bt_mom << endl;
      char nameHisto[256];
      sprintf(nameHisto,"bin_%d",(bt*nTbins_mom)+bt_mom);
      ReWeight[(bt*nTbins_mom)+bt_mom] = new TGraph(7);
      ReWeight[(bt*nTbins_mom)+bt_mom]->SetName(nameHisto);
      ReWeight[(bt*nTbins_mom)+bt_mom]->SetTitle(nameHisto);
      ReWeight[(bt*nTbins_mom)+bt_mom]->SetMarkerStyle(20);
      ReWeight[(bt*nTbins_mom)+bt_mom]->SetMarkerColor(2);
      for(int w=0;w<nWeights;w++){
        if(bt_mom==0 && bt==0) hreco[w]->Write(Form("recoHisto_weight_%d",w));
        //cout << "Current Histo:" << endl;
        //hreco[w]->Print("all");
        //cout << hreco[w]->Integral() << endl;
        if(hreco[w]->GetBinContent((bt*nTbins_mom)+bt_mom+1) !=0 ){
          ReWeight[(bt*nTbins_mom)+bt_mom]->SetPoint(w,MA[w],hreco[w]->GetBinContent((bt*nTbins_mom)+bt_mom+1)/hreco[3]->GetBinContent((bt*nTbins_mom)+bt_mom+1));
        }
        else{
          cout << "No events in bin " << endl;
          ReWeight[(bt*nTbins_mom)+bt_mom]->SetPoint(w,MA[w],1);
        }
        ReWeight[(bt*nTbins_mom)+bt_mom]->GetYaxis()->SetTitle("weight");
      }
      ReWeight[(bt*nTbins_mom)+bt_mom]->Write();
    }
  }  
  output->Close();
}
