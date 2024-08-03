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

  xsInputdataHighland2 *inpRun = new xsInputdataHighland2();
  //inpRun->SetupFromHighland(genieFD.c_str(),genieFD.c_str());   
  inpRun->SetupFromHighland(nuwroFD.c_str(),nuwroFD.c_str());   
  //inpRun->SetupFromHighland(neutMC.c_str(),neutMC.c_str());    

  inpRun->SetFlux("/data/t2k/dolan/xsToolBasedMEC/CC0PiAnl/fluxFiles/nd5_tuned11bv3.2_11anom_run2_fine.root", "enu_nd5_tuned11b_numu");
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
  
  //All sample signal
  inp->SetSignal("(mectopology==1 || mectopology==2) && ( (truep_truemom>450) && (truep_truemom<1000) && (truemu_mom>250) && (truemu_costheta>-0.6) && (truep_truecostheta>0.4) )");  


  inp->SetCut("Sum$(accum_level[xstool_throw][1]) > 8");

  //Don't use out of bounds bins:
  //inp->removeOutOfRange = 1;

  // define differential variables and binning
  std::cout << "define differential variables and binning " << std::endl;
  xsBinning *binningTru = new xsBinning();
  xsBinning *binningRec = new xsBinning();


  const int nbins = 8;
  const double bins[nbins+1] =   {0.0001, 0.08, 0.12, 0.155, 0.2, 0.26, 0.36, 0.51, 1.1};
  std::cout << "Adding binning ... " << std::endl;
  binningTru->addDimension(string("trueDpT"),nbins,bins);
  binningRec->addDimension(string("Sum$(recDpT[xstool_throw])"),nbins,bins);
  //binningRec->addDimension(string("Sum$(trueDphiT[xstool_throw])"),nbins,bins);
  std::cout << "Incorperating into input ... " << std::endl;


  std::cout << "Set Rec binning  ... " << std::endl;
  inp->SetReconstructedBinning(binningRec);
  std::cout << "Set True binning  ... " << std::endl;
  inp->SetTrueBinning(binningTru);

  // set cache file
  std::cout << "set cache file " << std::endl;
  inp->ReadCache("cache_cc0pidPUnfold.root");

  // create cross section engine for unfolding and feed with input data, flux and target mode
  std::cout << "create cross section engine" << std::endl;
  xsEngineUnfoldingBayes *engine = new xsEngineUnfoldingBayes(); engine->SetNumberOfIterations(1);
 // xsEngineNoop *engine = new xsEngineNoop();  // this would give you no cross-section but number of reconstructed events (not unfolded)

  engine->SetInput(inp);
  //engine->SetTargetMode(xsTargetSection::FGD1, xsTargetNucleons::NEUTRONS);  // protons and neutrons in FGD1 are targets
  //engine->SetNTargets(2.76906197720358541e+29);
  //engine->SetNTargets(2.75e29);
  engine->SetNTargets(5.5373e29);

  engine->SetEnableNominalReweighting(true);

  //Diable BG subtraction:
  engine->EnableBackgroundSubtraction(false);

  //Print some truth info: 
  cout <<"MC POT is: " << inp->GetPOTmc() << endl;
  cout <<"Data POT is: " << inp->GetPOTdata() << endl;

  (inp->GetReconstructed())->Print("all");
  (inp->GetTruth())->Print("all");

  //(engine->GetTruthResult())->Print("all");
  //cout << "Truth xsec is: " << engine->GetTruthTotalXsec() << endl;

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
  cout << "Total xsec = ( " << totalXS << " +- " << staterr << "(stat) +- " << systerr << "(syst) ) x 10^-38 cm^2 / Nucleon" << endl;
  cout << "Total xsec from truth = " << totalXStruth << " x 10^-38 cm^2 / Nucleon" << endl;

  // print error on total cross section for each error source
  for(int i=0; i<sources.size(); i++) {
    cout << sources[i] << "  abs: " << totalErrorPerSource[i] << "  rel: " << totalErrorPerSource[i]/totalXS*100. << "%" << endl;
  }
  cout << "************************************************************************" << endl;
  
  inp->SaveCache("cache_cc0pidPUnfold.root");
  
  
  /*********************************************************
  **** Actual cross section / error extraction end *********
  *********************************************************/
  
  TFile *outputf = new TFile("CC0pidPUnfoldOut.root","RECREATE");

  // draw unfolded cross section result
  TCanvas *cannix = new TCanvas();
  cannix->Divide(2);
  cannix->cd(1);
  gStyle->SetOptStat(1111);
  TH1 *hresult_syst = engine->GetResultWithErrors(sources);
  hresult_syst->GetXaxis()->SetTitle("p_{#mu}^{(unfolded)}");
  hresult_syst->GetYaxis()->SetTitleOffset(1.2);
  hresult_syst->GetYaxis()->SetTitle("d #sigma / d p_{#mu} ( 10^{-38} cm^{2} / GeV / Neutron )");
  hresult_syst->SetLineColor(kRed);
  hresult_syst->Draw();

  vector<string> staterrs;
  staterrs.push_back("statistics_mc");
  staterrs.push_back("statistics_data");
  TH1 *hresult = engine->GetResultWithErrors(staterrs);
  hresult->SetLineColor(kGreen);
  hresult->Draw("same");

  // draw result from truth into same canvas
  TH1 *htrue = engine->GetTruthResult();
  htrue->SetLineColor(kBlue);
  htrue->Draw("hist same");

  /* // use these lines to save the result, as it is needed for the validation macro for reference
  TFile *fout = new TFile("numuCCresult.root","RECREATE");
  fout->cd();
  hresult->Write("hresult");
  htrue->Write("htrue");
  fout->Close();
  // */
  
  // draw legend
  TLegend *leg = new TLegend(0.50,0.15,0.85,0.30);
  leg->AddEntry(hresult_syst,"MC reconstructed and unfolded","l");
  leg->AddEntry(hresult,"Statistical errors only","l");
  leg->AddEntry(htrue,"Calculated from MC truth","l");
  leg->Draw();

  // draw overall covariance matrix
  /*
  cannix->cd(2);
  TMatrixD *cvm = engine->GetCovarianceMultipleSources(sources);
  TH2D *hcvm = new TH2D(*cvm);
  double m = TMath::Max(hcvm->GetMaximum(),-hcvm->GetMinimum());
  hcvm->GetZaxis()->SetRangeUser(-m,m);
  hcvm->SetTitle("Covariance matrix for all systematic errors");
  hcvm->SetContour(nb);
  hcvm->Draw("colz");
  */

  cannix->Write("cannix");


  TCanvas *tutnix = new TCanvas();
  xsDrawingTools *draw = new xsDrawingTools(engine);
  if(reweightingOn == true) (draw->DrawResultWithGroupedErrors(xsErrorGroups::ORIGIN));
  else (draw->DrawResultWithGroupedErrors(xsErrorGroups::STATONLY));
  TLegend *leg = draw->GetLegend();

  // draw result from truth into same canvas
  //TH1 *htrue2 = binningTru->convertTH1(htrue);
  TH1 *htrue2 = binningTru->convertBinning(htrue);
  htrue2->SetLineColor(kBlue);
  htrue2->Draw("hist same");
  leg->AddEntry(htrue2,"NEUT (from truth)","l");
  //Write Canvas
  tutnix->Write("tutnix");
  htrue2->Write("truth");


  TCanvas *tutnixNoDet = new TCanvas();
  xsDrawingTools *draw = new xsDrawingTools(engine);
  if(reweightingOn == true) draw->DrawResultWithGroupedErrors(xsErrorGroups::NODETECTOR);
  //else (draw->DrawResultWithGroupedErrors(xsErrorGroups::STATONLY));
  TLegend *leg = draw->GetLegend();

  // draw result from truth into same canvas
  TH1 *htrue2 = binningTru->convertBinning(htrue);
  htrue2->SetLineColor(kBlue);
  htrue2->Draw("hist same");
  //leg->AddEntry(htrue2,"NEUT (from truth)","l");
  //Write Canvas
  tutnixNoDet->Write("tutnixNoDet");

/*
  // create another canvas to put sliced cross-section into
  TCanvas *tutnix = new TCanvas();
  tutnix->Divide(2,2);

  xsDrawingTools *draw = new xsDrawingTools(engine);
  draw->SetRangeX(0.,2.0);
  draw->SetRangeY(0.,1.8);

  gStyle->SetOptStat(0);
  for(int i=1; i<=9; i++) {
    tutnix->cd(i);
    
    vector<int> bb(2);
    bb[0] = i;

    draw->SetTitleX("p_{#mu} / GeV");
    draw->SetTitleY("#frac{d^{2} #sigma}{dp d cos #Theta} ( 10^{-38} cm^{2} / GeV / Nucleon )");
    draw->SetTitle(TString::Format("%4.2f < cos #Theta < %4.2f",Tbins[i],Tbins[i+1]).Data());
    draw->DrawResultWithGroupedErrors(xsErrorGroups::ORIGIN, 1,bb);
    
    TLegend *leg = draw->GetLegend();
    TH1 *hslice;

    hslice = binningTru->convertSliceTH1(htrue,1,bb);
    draw->ProperZoomX(hslice);  // properly zoom X axis with fractional bin
    hslice->SetLineColor(kBlue);
    hslice->Draw("same hist");
    leg->AddEntry(hslice,"NEUT (from truth)","l");

    hslice = binningTru->convertSliceTH1(hresult_syst,1,bb);
    draw->ProperZoomX(hslice);
    hslice->SetLineColor(kBlack);
    hslice->Draw("e0 same");
    leg->AddEntry(hslice,"Data (unfolded with NEUT)","l");
  }

  tutnix->Write("tutnix");
*/
//   // 3rd canvas, will contain measured and unfolded spectra as well as smearing and unsmearing matrices
//   TCanvas *wilnix = new TCanvas();
//   wilnix->Divide(2,2);
//   wilnix->cd(1);
//   engine->GetMeasured()->Draw();
//   wilnix->cd(2);
//   engine->GetUnfolded("nominal",0,false)->Draw();
//   wilnix->cd(3);
//   engine->GetSmearingMatrix()->Draw("colz");
//   wilnix->cd(4);
//   engine->GetUnsmearingMatrix()->Draw("colz");

//   wilnix->Write("wilnix");

//   // draw relative/fractional covariance matrices for each single source
//   TCanvas *wilnix2 = new TCanvas();
//   wilnix2->Divide(TMath::Ceil(sources.size()/3.),3);
//   for(int i=0; i<sources.size(); i++) {
//     wilnix2->cd(i+1);
//     TMatrixD *cvm = engine->GetCovariance(sources.at(i),true);
// /*
//     // uncomment this to draw error matrices (square-root of covariance matrix) instead
//     for(int k1=0; k1 < cvm->GetNcols(); k1++) for(int k2=0; k2 < cvm->GetNrows(); k2++) {
//       double val = (*cvm)[k1][k2];
//       val = val/TMath::Abs(val)*TMath::Sqrt(TMath::Abs(val));
//       (*cvm)[k1][k2] = val;
//     }
// */
//     TH2D *hcvm = new TH2D(*cvm);
//     hcvm->SetName(sources.at(i).c_str());
//     delete cvm;
//     double m = TMath::Max(hcvm->GetMaximum(),-hcvm->GetMinimum());
//     hcvm->GetZaxis()->SetRangeUser(-m,m);
//     hcvm->SetTitle(sources.at(i).c_str());
//     hcvm->SetContour(nb);
//     hcvm->Draw("colz");
//   }
//   wilnix2->Write("wilnix2");

//   // add absolute covariance matrices for all detector systematics
//   TCanvas *wilnix3 = new TCanvas();
//   wilnix3->cd();
//   TMatrixD *cvm_det = NULL;
//   for(int i=0; i<sources.size(); i++) {
//     if(sources[i].substr(0,9) != "highland_") continue;
//     TMatrixD *cvm = engine->GetCovariance(sources.at(i),false);
// /*
//     // uncomment this to draw error matrices (square-root of covariance matrix) instead
//     for(int k1=0; k1 < cvm->GetNcols(); k1++) for(int k2=0; k2 < cvm->GetNrows(); k2++) {
//       double val = (*cvm)[k1][k2];
//       val = val/TMath::Abs(val)*TMath::Sqrt(TMath::Abs(val));
//       (*cvm)[k1][k2] = val;
//     }
// */
//     if(cvm_det == NULL) {
//       cvm_det = cvm;
//     }
//     else {
//       *cvm_det += *cvm;
//       delete cvm;
//     }
//   }
//   if(cvm_det) {
//     TH2D *hcvm_det = new TH2D(*cvm_det);
//     double m = TMath::Max(hcvm_det->GetMaximum(),-hcvm_det->GetMinimum());
//     hcvm_det->GetZaxis()->SetRangeUser(-m,m);
//     hcvm_det->SetTitle("Detector systematics from Highland");
//     hcvm_det->SetContour(nb);
//     hcvm_det->Draw("colz");
//   }
//   wilnix3->Write("wilnix3");
  outputf->Close();
}
