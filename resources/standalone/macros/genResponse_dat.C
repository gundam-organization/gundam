//#include "genResponse_test.h"

bool DEBUGMODE=false;

void genResponse_dat(string microtreeIn, string rwpath, string dialName, double nom, double err, int topo, int reac, int dataType=0) {

  

  // declare variables for fit parameters, names, and values
  vector<string> param_names;
  vector<double> param_values;
  double bestfit;
  double error;
  //double sigma;


  // define inputs
  string microtree = microtreeIn;
  string rwfile_default = rwpath+"rwfile_default.root";
  string rwfile_truth = rwpath+"rwfile_truth.root";
  string flux = "/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/fluxFiles/nd5_tuned11bv3.2_11anom_run2_fine.root";
  string fluxname = "enu_nd5_tuned11b_numu";

  string outname = Form("topo%d_reac%d.root",topo,reac);
  string outputFileName = "./output/resFunc"+dialName+outname;
  cout << "Output file name is: " << outputFileName << endl; 
  
  
  // set up unfolding variables, signal, and cut
  string mc_variable = "trueDalphaT";
  string data_variable = "Sum$(recDalphaT[xstool_throw])";
  string signal, cut, bcut; // these will be defined in the main loop
  //string psCut = "&& (Sum$(selp_mom[xstool_throw])>450) && (Sum$(selmu_mom[xstool_throw])>150) && (Sum$(selp_theta[xstool_throw])>-3.15) && (Sum$(selmu_theta[xstool_throw])>1.57)";

  //Phase Space Bins:
  const int npmombins=1;
  const double pmombins[npmombins+1] = {450.0, 1000.0};
  const int npthetabins=1;
  const double pthetabins[npthetabins+1] = {0.4, 1.0};
  const int nmumombins=1;
  const double mumombins[npmombins+1] = {250.0, 10000.0};
  const int nmuthetabins=1;
  const double muthetabins[npthetabins+1] = {-0.6, 1.0};

  // set up recon binning
  const int nRbins = 8;
  const double Rbins[nRbins+1] = { 0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};


  // set up true binning
  const int nTbins = 8;
  const double Tbins[nTbins+1] = { 0.00001, 0.47, 1.02, 1.54, 1.98, 2.34, 2.64, 2.89, 3.14159};


  //Extra PS bins, not binned in var of interest
  const int nextrabins=1;
  double Ebins[nextrabins][8];

  //Currently these extra bins (for multiple OOPS bins split my proton/muon kinematics) are not used
  /*
  const int nextrabins=5;
  double Ebins[nextrabins][8];
  //                Proton Mom      Proton Ang  Muon Mom       Mu Ang
  double Ebins0[8]={0.0,450.0,      -1.0,0.4,   0.0,10000.0,   -1.0, 1.0 }; // low mom, high angle protons
  double Ebins1[8]={0.0,450.0,      0.4,1.0,    0.0,10000.0,   -1.0, 1.0 }; // low mom, forward protons
  double Ebins2[8]={450.0,1000.0,   -1.0,0.4,   0.0,10000.0,   -1.0, 1.0 }; // good mom, high angle protons
  double Ebins3[8]={450.0,1000.0,   0.4,1.0,    0.0,250.0,     -1.0, 1.0 }; // good mom, forward protons + low mom muons
  double Ebins4[8]={1000.0,10000.0, -1.0,1.0,   250.0,10000.0, -1.0, 1.0}; // high mom protons
  for(int j=0; j<8; j++){
    Ebins[0][j]=Ebins0[j]; Ebins[1][j]=Ebins1[j]; Ebins[2][j]=Ebins2[j]; Ebins[3][j]=Ebins3[j]; Ebins[4][j]=Ebins4[j];
  }
  */
  //                Proton Mom      Proton Ang  Muon Mom       Mu Ang
  double Ebins0[8]={450.0,1000.0,      0.4,1.0,   250.0,10000.0,   -0.6, 1.0 }; // signal region
  for(int j=0; j<8; j++){ Ebins[0][j]=Ebins0[j];}


  //DEBUG binning:
  // const int nRbins = 1;
  // const double Rbins[nRbins+1] = { 0.0, 0.08};
  // const int nTbins = 1;
  // const double Tbins[nTbins+1] = { 0.0, 0.08};
  if(DEBUGMODE){
    TH1D* dummy = new TH1D("dummy", "dummy", 1, 0, 0.08);
    dummy->Fill(0.04);
  }

  // Set up hist for global bin mapping:
  const int nTotalBins=nTbins+nextrabins;
  //TH3D* binMapHist = new TH3D("binMapHist","binMapHist",nTbins,Tbins,npmombins,pmombins,npthetabins,pthetabins);


  // You need to provide the number of branches in your HL2 tree
  // And the accum_level you want to cut each one at to get your selected events
  // i.e choosing n in accum_level[0][branch_i]>n
  if(dataType==0){
    const int nbranches = 10;
    const int accumToCut[nbranches] =   {7,8,9,8,7,5,4,7,8,7};
  }  
  else{
    const int nbranches = 1;
    const int accumToCut[nbranches] = {0};
  }  

  
  // delcare xsBinning, they will be used always the same in the main loop
  xsBinning  *binningTru = new xsBinning();
  xsBinning  *binningRec = new xsBinning();
  binningTru->addDimension(mc_variable,nTbins,Tbins);
  binningRec->addDimension(data_variable,nRbins,Rbins);
  
  // setup on-the-fly reweighting parameter
  param_names.push_back(dialName);
  bestfit = nom;
  error = err;

  // Total number of topologies
  const int nTopologies = nbranches;
  // Total number of weights
  //const int nWeights = 4;
  const int nWeights = 7;
  // reweighted reconstrcted distribution for each weight, true bin, and topology
  TH1 *hreco[nWeights][nTotalBins][nTopologies];


  int globalCount=0;
  for(const int bt = 0; bt < nTbins; bt++){
    if(DEBUGMODE) break;
    int t = topo;
    int r = reac;        
    if(t==0 || t==4 || r==6) continue; // Ignore branches with no proton

    cout << "Current bins (global dat): " << globalCount << " " << bt << endl << endl;
    for(const int w = 0; w < nWeights; w++){

      // define xsInput
      xsInputdataHighland2 *inpRwFly = new xsInputdataHighland2();
      inpRwFly->SetupFromHighland(microtree,microtree);
      inpRwFly->SetFlux(flux, fluxname);
      inpRwFly->SetRooTrackerVtxFiles(rwfile_default,rwfile_truth);
      inpRwFly->SetTrueBinning(binningTru);
      inpRwFly->SetReconstructedBinning(binningRec);

      // set cut definition, for each topology
      cut = Form("((Sum$(accum_level[xstool_throw][%d]) > %d) && (mectopology==%d))", t, accumToCut[t], r);

      // set signal definition, for each true bin
      //signal = Form("(trueDpT) > %f && (trueDpT) < %f  && (truep_truemom) > %f && (truep_truemom) < %f && (truep_truecostheta) > %f && (truep_truecostheta) < %f  && (truemu_mom) > %f && (truemu_mom) < %f && (truemu_costheta) > %f && (truemu_costheta) < %f",Tbins[bt],Tbins[bt+1],pmombins[0],pmombins[1],pthetabins[0],pthetabins[1],mumombins[0],mumombins[1],muthetabins[0],muthetabins[1]);
      signal = ("truemu_mom>0");

      // Account for no OOPS binning for SB regions:
      if(t==5 || t==6) bcut = Form(  " ( (Sum$(trueDalphaT[xstool_throw]) > %f) && (Sum$(trueDalphaT[xstool_throw])) < %f )  ", Tbins[bt],Tbins[bt+1]);
      else bcut = Form( "((Sum$(trueDalphaT[xstool_throw]) > %f) && (Sum$(trueDalphaT[xstool_throw])) < %f) && ((Sum$(truep_truemom[xstool_throw]) > %f) && (Sum$(truep_truemom[xstool_throw])) < %f) && ((Sum$(truep_truecostheta[xstool_throw]) > %f) && (Sum$(truep_truecostheta[xstool_throw])) < %f) && ((Sum$(truemu_mom[xstool_throw]) > %f) && (Sum$(truemu_mom[xstool_throw])) < %f) && ((Sum$(truemu_costheta[xstool_throw]) > %f) && ((Sum$(truemu_costheta[xstool_throw])) < %f)) ", Tbins[bt],Tbins[bt+1],pmombins[0],pmombins[1],pthetabins[0],pthetabins[1],mumombins[0],mumombins[1],muthetabins[0],muthetabins[1]);


      inpRwFly->SetSignal(signal.c_str());
      inpRwFly->SetCut((cut+" && "+bcut).c_str());

      // define xsReweightParameter
      xsReweightParameter *rwPar = new xsReweightParameter(param_names);
      inpRwFly->AddReweighting(rwPar);


      // set parameter value
      param_values.clear();    
      //param_values.push_back((-3+w)*error/bestfit);
      param_values.push_back((bestfit-1)+((-3+w)*error));
      string name = rwPar->SetParameterValue(param_values);

      // reweighting with the given value
      if(!DEBUGMODE) inpRwFly->GenerateThrow(name,0,true);

      // get reconstructed distribution
      hreco[w][globalCount][t] = inpRwFly->GetReconstructed();
      cout << "Reconstructed histo: w, b, t: " << w << ", " << globalCount << ", " << t << endl; 
      hreco[w][globalCount][t]->Print("all");
      if(DEBUGMODE){
        hreco[w][globalCount][t] = dummy;
        hreco[w][globalCount][t]->Print();
      }

      // delete xsInput
      delete inpRwFly;
    }
    // Increment the global bin counter
    globalCount++;
  }
  for(const int be = 0; be < nextrabins; be++){
    int t = topo;
    int r = reac;        
    if(t==0 || t==4 || r==6) continue; // Ignore branches with no proton
    cout << "Current bins (global extre): " << globalCount << " " << be << endl << endl;
    for(const int w = 0; w < nWeights; w++){

      // define xsInput
      xsInputdataHighland2 *inpRwFly = new xsInputdataHighland2();
      inpRwFly->SetupFromHighland(microtree,microtree);
      inpRwFly->SetFlux(flux, fluxname);
      inpRwFly->SetRooTrackerVtxFiles(rwfile_default,rwfile_truth);
      inpRwFly->SetTrueBinning(binningTru);
      inpRwFly->SetReconstructedBinning(binningRec);
    
      // set cut definition, for each topology
      cut = Form("((Sum$(accum_level[xstool_throw][%d]) > %d) && (mectopology==%d))", t, accumToCut[t], r);
    
      // set signal definition, for each true bin
      //signal = Form("(truep_truemom) > %f && (truep_truemom) < %f && (truep_truecostheta) > %f && (truep_truecostheta) < %f  && (truemu_mom) > %f && (truemu_mom) < %f && (truemu_costheta) > %f && (truemu_costheta) < %f",Ebins[be][0],Ebins[be][1],Ebins[be][2],Ebins[be][3],Ebins[be][4],Ebins[be][5],Ebins[be][6],Ebins[be][7]);
      signal = ("truemu_mom>0");

      //This definition is for many OOPS bins in proton/muon kinematics
      //bcut = Form( "((Sum$(truep_truemom[xstool_throw]) > %f) && (Sum$(truep_truemom[xstool_throw])) < %f) && ((Sum$(truep_truecostheta[xstool_throw]) > %f) && (Sum$(truep_truecostheta[xstool_throw])) < %f) && ((Sum$(truemu_mom[xstool_throw]) > %f) && (Sum$(truemu_mom[xstool_throw])) < %f) && ((Sum$(truemu_costheta[xstool_throw]) > %f) && ((Sum$(truemu_costheta[xstool_throw])) < %f))",Ebins[be][0],Ebins[be][1],Ebins[be][2],Ebins[be][3],Ebins[be][4],Ebins[be][5],Ebins[be][6],Ebins[be][7]);
    
      //This definition is for single OOPS bin which inverts the signal PS constraints
      bcut = Form( "((Sum$(truep_truemom[xstool_throw]) < %f) || (Sum$(truep_truemom[xstool_throw])) > %f) || ((Sum$(truep_truecostheta[xstool_throw]) < %f) || (Sum$(truep_truecostheta[xstool_throw])) > %f) || ((Sum$(truemu_mom[xstool_throw]) < %f) || (Sum$(truemu_mom[xstool_throw])) > %f) || ((Sum$(truemu_costheta[xstool_throw]) < %f) || ((Sum$(truemu_costheta[xstool_throw])) > %f))",Ebins[be][0],Ebins[be][1],Ebins[be][2],Ebins[be][3],Ebins[be][4],Ebins[be][5],Ebins[be][6],Ebins[be][7]);

      inpRwFly->SetSignal(signal.c_str());
      inpRwFly->SetCut(("("+cut+") && ("+bcut+")").c_str());
    
      // define xsReweightParameter
      
      xsReweightParameter *rwPar = new xsReweightParameter(param_names);
      inpRwFly->AddReweighting(rwPar);
  
      // set parameter value
      param_values.clear();    
      //param_values.push_back((-3+w)*error/bestfit);
      param_values.push_back((bestfit-1)+((-3+w)*error));
      string name = rwPar->SetParameterValue(param_values);
    
      // reweighting with the given value
      inpRwFly->GenerateThrow(name,0,true);
    
      // get reconstructed distribution
      hreco[w][globalCount][t] = inpRwFly->GetReconstructed();
      cout << "Reconstructed histo: w, b, t: " << w << ", " << globalCount << ", " << t << endl; 
      hreco[w][globalCount][t]->Print("all");
      if(DEBUGMODE){
        hreco[w][globalCount][t] = dummy;
        hreco[w][globalCount][t]->Print();
      }

      // delete xsInput
      delete inpRwFly;    }
    // Increment the global bin counter
    globalCount++;
  }

  
    
  //  }

  cout << "Total number of bins is: " << globalCount << endl;
  //Write TGraph in the output file
  TFile *output = new TFile(outputFileName.c_str(),"RECREATE");   
  
  for(const int t = 0; t < nTopologies; t++){  
    output->mkdir(Form("topology_%d",t));
  }
  char dir[200];

  //TGraph *ReWeight[nTotalBins][nTotalBins][nTopologies];
  TGraph *ReWeight[nTotalBins][nTopologies];
  
  double MA[7];
  for(int w = 0; w < 7; w++){
      MA[w]=bestfit-error*(3-w);
  }  

  //for(int br = 0; br<globalCount; br++){//reco kinematics bin
    for(int bt = 0; bt < globalCount; bt++){//true kinematics bin
      //if(fabs(br-bt)>20) continue;  //save memory if reco bin and true bin very far away
      //cout << "On true bin and reco bin: " << br << " and " << bt << endl;
      cout << "On true bin " << bt << endl;
      t=topo;
      r=reac;
      sprintf(dir,"topology_%d",t);
      output->cd(dir);
      char nameHisto[256];
      //sprintf(nameHisto,"RecBin_%d_trueBin_%d_topology_%d_reac_%d",br,bt,t,r);
      sprintf(nameHisto,"trueBin_%d_topology_%d_reac_%d",bt,t,r);
      ReWeight[bt][t] = new TGraph(7);
      ReWeight[bt][t]->SetName(nameHisto);
      ReWeight[bt][t]->SetTitle(nameHisto);
      ReWeight[bt][t]->SetMarkerStyle(20);
      ReWeight[bt][t]->SetMarkerColor(2);
      for(int w=0;w<nWeights;w++){
        if(DEBUGMODE){
          cout << "w, bt, t: " << w << ", " << bt << ", " << t << endl;
          hreco[w][bt][t]->Print();
          cout << "Histo printed" << endl;
        }
        hreco[w][bt][t]->Write(Form("recoHisto_weight_%d_trueBin_%d_topology_%d_reac_%d",w,bt,t,r));
        cout << "Current Histo:" << endl;
        hreco[w][bt][t]->Print("all");
        cout << hreco[w][bt][t]->Integral() << endl;
        //if(hreco[w][bt][t]->GetBinContent(br+1) !=0 ){
        if(hreco[w][bt][t]->Integral() !=0 ){
          //ReWeight[br][bt][t]->SetPoint(w,MA[w],hreco[w][bt][t]->GetBinContent(br+1)/hreco[3][bt][t]->GetBinContent(br+1));
          ReWeight[bt][t]->SetPoint(w,MA[w],hreco[w][bt][t]->Integral()/hreco[3][bt][t]->Integral());
        }
        else{
          cout << "No events in true bin " << endl;
          ReWeight[bt][t]->SetPoint(w,MA[w],1);
        }
        ReWeight[bt][t]->GetYaxis()->SetTitle("weight");
      }
      ReWeight[bt][t]->Write();
    }
  //}
  
  output->Close();
  
}
