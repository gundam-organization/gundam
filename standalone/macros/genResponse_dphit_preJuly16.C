//#include "genResponse_test.h"

bool DEBUGMODE=false;

void genResponse_dphit(string microtreeIn, string rwpath, string dialName, double nom, double err, int topo, int reac, int dataType=0) {

  

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
  string mc_variable = "trueDphiT";
  string data_variable = "Sum$(recDphiT[xstool_throw])";
  string signal, cut, bcut; // these will be defined in the main loop
  //string psCut = "&& (Sum$(selp_mom[xstool_throw])>450) && (Sum$(selmu_mom[xstool_throw])>150) && (Sum$(selp_theta[xstool_throw])>-3.15) && (Sum$(selmu_theta[xstool_throw])>1.57)";

  //Phase Space Bins:
  const int npmombins=3;
  const double pmombins[npmombins+1] = {0.0, 450.0, 1000.0, 2000.0};
  const int npthetabins=2;
  const double pthetabins[npthetabins+1] = {-1.0, 0.4, 1.0};

  // set up recon binning
  const int nRbins = 8;
  const double Rbins[nRbins+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};


  // set up true binning
  const int nTbins = 8;
  const double Tbins[nTbins+1] = { 0.0, 0.067, 0.14, 0.225, 0.34, 0.52, 0.85, 1.5, 3.14159};


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
  const int nTotalBins=nTbins*npthetabins*npmombins;
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
  xsBinning *binningTru = new xsBinning();
  xsBinning *binningRec = new xsBinning();
  binningTru->addDimension(mc_variable,nTbins,Tbins);
  binningRec->addDimension(data_variable,nRbins,Rbins);
  
  // setup on-the-fly reweighting parameter
  param_names.push_back(dialName);
  bestfit = nom;
  error = err;

  // Total number of topologies
  const int nTopologies = nbranches;
  // Total number of weights
  const int nWeights = 7;
  // reweighted reconstrcted distribution for each weight, true bin, and topology
  TH1 *hreco[nWeights][nTotalBins][nTopologies];

  int globalCount=0;
  for(const int bt = 0; bt < nTbins; bt++){
    for(const int bpm = 0; bpm < npmombins; bpm++){
      for(const int bpt = 0; bpt < npthetabins; bpt++){
        if(bpm==2 && bpt==0) continue; // MC stats are too low for this combination (high angle high mom protons)
        int t = topo;
        int r = reac;        
        if(t==0 || t==4 || r==6) continue; // Ignore branches with no proton
        cout << "Current bins (global DphiT protonMom protonAngle): " << globalCount << " " << bt << " " << bpm << " " << bpt << endl << endl;
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
          signal = Form("(trueDphiT) > %f && (trueDphiT) < %f  && (truep_truemom) > %f && (truep_truemom) < %f && (truep_truecostheta) > %f && (truep_truecostheta) < %f", Tbins[bt],Tbins[bt+1],pmombins[bpm],pmombins[bpm+1],pthetabins[bpt],pthetabins[bpt+1]);
          bcut = Form("((Sum$(trueDphiT[xstool_throw]) > %f) && (Sum$(trueDphiT[xstool_throw])) < %f) && ((Sum$(truep_truemom[xstool_throw]) > %f) && (Sum$(truep_truemom[xstool_throw])) < %f) && ((Sum$(truep_truecostheta[xstool_throw]) > %f) && (Sum$(truep_truecostheta[xstool_throw])) < %f)", Tbins[bt],Tbins[bt+1],pmombins[bpm],pmombins[bpm+1],pthetabins[bpt],pthetabins[bpt+1]);

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
    }
  }
  
    
  //  }

  cout << "Total number of bins is: " << globalCount << endl;
  //Write TGraph in the output file
  TFile *output = new TFile(outputFileName.c_str(),"RECREATE");   
  
  for(const int t = 0; t < nTopologies; t++){  
    output->mkdir(Form("topology_%d",t));
  }
  char dir[200];

  TGraph *ReWeight[nTotalBins][nTotalBins][nTopologies];
  
  double MA[7];
  for(int w = 0; w < 7; w++){
      MA[w]=bestfit-error*(3-w);
  }  

  for(int br = 0; br<globalCount; br++){//reco kinematics bin
    for(int bt = 0; bt < globalCount; bt++){//true kinematics bin
      //if(fabs(br-bt)>20) continue;  //save memory if reco bin and true bin very far away
      cout << "On true bin and reco bin: " << br << " and " << bt << endl;
      t=topo;
      r=reac;
      sprintf(dir,"topology_%d",t);
      output->cd(dir);
      char nameHisto[256];
      sprintf(nameHisto,"RecBin_%d_trueBin_%d_topology_%d_reac_%d",br,bt,t,r);
      ReWeight[br][bt][t] = new TGraph(7);
      ReWeight[br][bt][t]->SetName(nameHisto);
      ReWeight[br][bt][t]->SetTitle(nameHisto);
      ReWeight[br][bt][t]->SetMarkerStyle(20);
      ReWeight[br][bt][t]->SetMarkerColor(2);
      for(int w=0;w<nWeights;w++){
        if(DEBUGMODE){
          cout << "w, bt, t: " << w << ", " << bt << ", " << t << endl;
          hreco[w][bt][t]->Print();
          cout << "Histo printed" << endl;
        }
        if(hreco[w][bt][t]->GetBinContent(br+1) !=0 ){
          ReWeight[br][bt][t]->SetPoint(w,MA[w],hreco[w][bt][t]->GetBinContent(br+1)/hreco[3][bt][t]->GetBinContent(br+1));
        }
        else{
          ReWeight[br][bt][t]->SetPoint(w,MA[w],1);
        }
        ReWeight[br][bt][t]->GetYaxis()->SetTitle("weight");
      }
      ReWeight[br][bt][t]->Write();
    }
  }
  
  output->Close();
  
}
