/**
      @file
      @author Martin Hierholzer <martin.hierholzer@lhep.unibe.ch>
      
      @section DESCRIPTION

      Generic tool to extract differential cross sections from ND280 data
*/

const bool enableRealData = false;
const bool mergeDetectorErrors = false;         // not needed if we use all_syst, which we should anyway!
const bool rmErr = true;
const bool rmAllErr = false;
const bool reweightingOn = false;

/// Analysis ROOT macro for the numu CC inclusive tracker analysis.
void cctransdphi() {
  //gSystem->Load("libxsTool.so");

  // set histogram style and create canvas
  gStyle->SetOptStat(0);
  gStyle->SetTitleOffset(1.2,"X");
  gStyle->SetTitleOffset(2.0,"Y");
  gStyle->SetPadRightMargin(0.15);

  const UInt_t Number = 5;
  Double_t Red[Number]    = { 0.00, 0.00, 1.00, 1.00, 1.00 };
  Double_t Green[Number]  = { 0.00, 1.00, 1.00, 1.00, 0.00 };
  Double_t Blue[Number]   = { 1.00, 1.00, 1.00, 0.00, 0.00 };
  Double_t Length[Number] = { 0.00, 0.40, 0.50, 0.60, 1.00 };
  Int_t nb=50;
  TColor::CreateGradientColorTable(Number,Length,Red,Green,Blue,nb);
  
  /*********************************************************
  **** Actual cross section / error extraction start *******
  *********************************************************/
  

  // prepare input data 
  std::cout << "prepare input data" << std::endl;

  string neutMC="/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_NeutAirAllSystV5_out/allMerged.root";
  string neutFD="/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_NeutWaterNoSystV2_out/allMerged.root";
  string genieFD="/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_GenieAirNoSystV2_out/allMerged.root";
  string nuwroFD="/data/t2k/dolan/MECProcessing/CC0Pi/mar15HL2/job_NuwroAllSystV1_out/allMerged.root";

  string neutMC_jan17="/data/t2k/dolan/MECProcessing/CC0Pi/feb17HL2/job_NeutAirV3_pFromRoo_out/allMerged.root";


  xsInputdataHighland2 *inpRun = new xsInputdataHighland2();
  //inpRun->SetupFromHighland(genieFD.c_str(),genieFD.c_str());   
  //inpRun->SetupFromHighland(nuwroFD.c_str(),nuwroFD.c_str()); inpRun->SetPOT(4.011E21,4.011E21,4.011E21);
  inpRun->SetupFromHighland(neutMC_jan17.c_str(),neutMC_jan17.c_str());  

  //inpRun->SetupFromHighland(neutMC_jan17.c_str(),neutMC_jan17.c_str());  
  

  inpRun->SetFlux("/data/t2k/dolan/fitting/xsecFromMC/nd5_tuned13av1.1_13anom_run1-5b.root", "enu_nd5_13a_nom_numu");
  if(mergeDetectorErrors) inpRun->SetMergeDetectorErrors();
  //inpRun->SetMergeDetectorErrors(); // Only doing this since test microtree doesn't have detector errors
  
  // setup reweighting  
  std::cout << "setup reweighting " << std::endl;
  //xsReweightSystematics *reweightRun = new xsReweightSystematics("/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/weights/unifiedWeightsFiles/cc0piv7_interactive.root","numu");
  if (reweightingOn==true){
    xsReweightSystematics *reweightRun = new xsReweightSystematics("/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/weights/outdir/cc0piv27Mar16NeutAirV2/weights_unified.root","numu");
    inpRun->AddReweighting(reweightRun);
  }

  // merge all runs
  std::cout << "merge all runs " << std::endl;
  xsInputdataMultipleRuns *inp = new xsInputdataMultipleRuns();
  inp->AddRun(inpRun);
  
  // define signal and selection cut
  std::cout << "define signal and selection cut " << std::endl;
  //inp->SetSignal("(mectopology==1 || mectopology==2) && ( (truep_truemom>450) && (truep_truemom<1000) && (truemu_mom>250) && (truemu_costheta>-0.6) && (truep_truecostheta>0.4) )");  
  inp->SetSignal("(mectopology==1 || mectopology==2) && ( (truep_truemom>450) && (truep_truemom<1000) && (truelepton_mom>250) && (truelepton_costheta>-0.6) && (truep_truecostheta>0.4) )");  
  //inp->SetSignal("(mectopology<3) && ((truep_truemom>450) && (truep_truemom<1000))");  
  //inp->SetSignal("reactionCC==1");  
 
  inp->SetCut("Sum$(accum_level[xstool_throw][1]) > 8");

  //Don't use out of bounds bins:

  //inp->removeOutOfRange = 1;

  // define differential variables and binning
  std::cout << "define differential variables and binning " << std::endl;
  xsBinning *binningTru = new xsBinning();
  xsBinning *binningRec = new xsBinning();


  const int nbins = 8;
  const double bins[nbins+1] =   { 0.0, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
  std::cout << "Adding binning ... " << std::endl;
  binningTru->addDimension(string("trueDpT"),nbins,bins);
  binningRec->addDimension(string("Sum$(recDpT[xstool_throw])"),nbins,bins);
  //binningRec->addDimension(string("Sum$(trueDphiT[xstool_throw])"),nbins,bins);
  std::cout << "Incorperating into input ... " << std::endl;


  std::cout << "Set Rec binning  ... " << std::endl;
  inp->SetReconstructedBinning(binningRec);
  std::cout << "Set True binning  ... " << std::endl;
  inp->SetTrueBinning(binningTru);

  // create cross section engine for unfolding and feed with input data, flux and target mode
  std::cout << "create cross section engine" << std::endl;
  xsEngineUnfoldingBayes *engine = new xsEngineUnfoldingBayes(); engine->SetNumberOfIterations(1);

  engine->SetInput(inp);
//engine->SetTargetMode(xsTargetSection::FGD1, xsTargetNucleons::BOTH);  // protons and neutrons in FGD1 are targets
  engine->SetNTargets(5.5373e29); // protons and neutrons in FGD1 are targets

  engine->SetEnableNominalReweighting(true);

  //Diable BG subtraction:
  engine->EnableBackgroundSubtraction(false);

  (inp->GetReconstructed())->Print("all");
  (inp->GetTruth())->Print("all");

  // Note: all calculations are done "on the fly" when calling the getter-functions like GetResult()
  
  // obtain list of error sources
  std::cout << "obtain list of error sources:" << std::endl;
  vector<string> sources = inp->GetErrorSources();
  vector<string>::iterator errIt;
  for(errIt=sources.begin(); errIt!=sources.end(); ++errIt){
    std::cout << "source: " << (*errIt) << std::endl;
  }
  //Temp hack to remove detector errors: 
  if(rmErr==true){
    std::cout << "removing det errors:" << std::endl;
    sources.erase(sources.end());
    for(errIt=sources.begin(); errIt!=sources.end(); ++errIt){
     std::cout << "source: " << (*errIt) << std::endl;
    }
  }
  if(rmAllErr==true){
    std::cout << "removing all errors:" << std::endl;
    sources.erase(sources.end());
    sources.erase(sources.end());
    sources.erase(sources.end());
    
    //for(errIt=sources.begin(); errIt!=sources.end(); ++errIt){
    // std::cout << "source: " << (*errIt) << std::endl;
    //}
  }


  // calculate error on total cross section for each error source
  std::cout << "calculate error on total cross section for each error source" << std::endl;
  vector<double> totalErrorPerSource;
  double staterr = 0;
  double systerr = 0;
  for(int i=0; i<sources.size(); i++) {
    double err = engine->GetTotalXsecError(sources[i]);
    totalErrorPerSource.push_back(err);
    if(sources[i].substr(0,strlen("statistics_")) == "statistics_") {
      staterr += pow(err,2);
    }
    else {
      systerr += pow(err,2);
    }
  }
  staterr = sqrt(staterr);
  systerr = sqrt(systerr);

  // print out the total cross-section
  double totalXS = engine->GetTotalXsec();
  double totalXStruth = engine->GetTotalXsecFromTruth();
  cout << "************************************************************************" << endl;
  // cout << "Total xsec = ( " << totalXS << " +- " << staterr << "(stat) +- " << systerr << "(syst) ) x 10^-38 cm^2 / Nucleon" << endl;
  cout << "Total xsec from truth = " << totalXStruth << " x 10^-38 cm^2 / Nucleon" << endl;

  // // print error on total cross section for each error source
  // for(int i=0; i<sources.size(); i++) {
  //   cout << sources[i] << "  abs: " << totalErrorPerSource[i] << "  rel: " << totalErrorPerSource[i]/totalXS*100. << "%" << endl;
  // }
  cout << "************************************************************************" << endl;
    
  
  /*********************************************************
  **** Actual cross section / error extraction end *********
  *********************************************************/
  
  TFile *outputf = new TFile("dpt_MCTruth.root","RECREATE");

  TH1 *htrue = engine->GetTruthResult();
  htrue->SetLineColor(kBlue);
  TH1 *htrue2 = binningTru->convertBinning(htrue);
  htrue2->SetLineColor(kBlue);
  htrue2->Scale(1E-38);
  htrue2->Write("truth");
  
  outputf->Close();
}
