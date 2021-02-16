#include "ND280Fitter.hh"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"
#include "Logger.h"
#include <TCanvas.h>
#include <THStack.h>
#include <iostream>
#include <GlobalVariables.h>

#define FIT_IT_TIME_POINT 10
#define REWEIGHT_TIME_POINT 11
#define FILL_TIME_POINT 12


ND280Fitter::ND280Fitter() {

    Logger::setUserHeaderStr("[ND280Fitter]");

    // Memory allocation
    _PRNG_ = new TRandom3();
    gRandom = _PRNG_;

    // Nullifying Native Objects which are handled by this class
    _minimizer_ = nullptr;
    _functor_   = nullptr;

    Reset();

}

ND280Fitter::~ND280Fitter()
{
    Reset();

    delete _PRNG_; // deleting a nullptr has no effect anyway

}

void ND280Fitter::Reset(){

    _is_initialized_ = false;

    if(GlobalVariables::getNbThreads() > 1 and _stopThreads_ == false){
        LogInfo << "Stopping parallel threads..." << std::endl;
        _stopThreads_ = true;
        // the following provoques Segfault
//        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads() ; iThread++ ){
//            _asyncFitThreads_[iThread].get(); // Waiting for the end of each thread
//        }
    }

    _fitHasBeenDone_    = false;
    _fit_has_converged_ = false;

    _PRNG_seed_ = -1;

    delete _minimizer_;         _minimizer_ = nullptr;
    delete _functor_;           _functor_ = nullptr;

    _outputTDirectory_ = nullptr; // not handle by this class

    _save_fit_params_   = false;
    _save_event_tree_   = true;
    _disableChi2Pulls_                          = false;
    _apply_statistical_fluctuations_on_samples_ = false;

    _saveFitParamsFrequency_     = 10000;
    _nb_fit_parameters_          = 0;
    _nbFitCalls_                 = 0;
    _nbTotalEvents_              = 0;

    _MC_normalization_factor_ = 1;

    _selected_data_type_ = DataType::kReset;

    // Default Minimization Settings
    _minimization_settings_.minimizer   = "Minuit2";
    _minimization_settings_.algorithm   = "Migrad";
    _minimization_settings_.print_level = 2;
    _minimization_settings_.strategy    = 1;
    _minimization_settings_.tolerance   = 1E-2;
    _minimization_settings_.max_iter    = 1E6;
    _minimization_settings_.max_fcn     = 1E9;

    _fitParametersGroupList_.clear();
    _samplesList_.clear();

    _parameter_names_.clear();
    _parameterPriorValues_.clear();

}

void ND280Fitter::SetOutputTDirectory(TDirectory* output_tdirectory_) {
    _outputTDirectory_ = output_tdirectory_;
}
void ND280Fitter::SetPrngSeed(int PRNG_seed_){
    _PRNG_seed_ = PRNG_seed_;
}
void ND280Fitter::SetMcNormalizationFactor(double MC_normalization_factor_){
    _MC_normalization_factor_ = MC_normalization_factor_;
}
void ND280Fitter::SetMinimizationSettings(const MinSettings& minimization_settings_){
    _minimization_settings_ = minimization_settings_;
    if(_minimizer_ != nullptr){
        _minimizer_->SetStrategy(_minimization_settings_.strategy);
        _minimizer_->SetPrintLevel(_minimization_settings_.print_level);
        _minimizer_->SetTolerance(_minimization_settings_.tolerance);
        _minimizer_->SetMaxIterations(_minimization_settings_.max_iter);
        _minimizer_->SetMaxFunctionCalls(_minimization_settings_.max_fcn);
    }
}
void ND280Fitter::SetDisableSystFit(bool disable_syst_fit_) {
    _disableChi2Pulls_ = disable_syst_fit_;
}
void ND280Fitter::SetSaveEventTree(bool save_event_tree_){
    _save_event_tree_ = save_event_tree_;
}
void ND280Fitter::SetSaveFitParams(bool save_fit_params_){
    _save_fit_params_ = save_fit_params_;
}
void ND280Fitter::SetApplyStatisticalFluctuationsOnSamples(bool apply_statistical_fluctuations_on_samples_){
    _apply_statistical_fluctuations_on_samples_ = apply_statistical_fluctuations_on_samples_;
}
void ND280Fitter::SetSaveFitParamsFrequency(int save_fit_params_frequency_){
    _saveFitParamsFrequency_ = save_fit_params_frequency_;
}
void ND280Fitter::SetAnaFitParametersList(std::vector<AnaFitParameters*> AnaFitParameters_list_){
    _fitParametersGroupList_ = std::move(AnaFitParameters_list_);
}
void ND280Fitter::SetAnaSamplesList(std::vector<AnaSample*> AnaSample_list_){
    _samplesList_   = AnaSample_list_;
    _nbTotalEvents_ = 0;
    for(const auto& anaSample : _samplesList_){
        _nbTotalEvents_ += anaSample->GetN();
    }
}
void ND280Fitter::SetSelectedDataType(int selected_data_type_){
    _selected_data_type_ = static_cast<DataType>(selected_data_type_);
}


void ND280Fitter::Initialize(){

    LogWarning << "ND280 Fitter is initializing..." << std::endl;
    PassIfNotInitialized(__METHOD_NAME__);

    if(_outputTDirectory_ == nullptr){
        LogFatal << "_outputTDirectory_ has not been set." << std::endl;
        throw std::logic_error("_outputTDirectory_ has not been set.");
    }
    else if(_PRNG_seed_ == -1){
        LogFatal << "_PRNG_seed_ has not been set." << std::endl;
        throw std::logic_error("_PRNG_seed_ has not been set.");
    }
    else if(_samplesList_.empty()){
        LogFatal << "_samplesList_ has not been set." << std::endl;
        throw std::logic_error("_samplesList_ has not been set.");
    }
    else if(_selected_data_type_ == kReset){
        LogFatal << "_selected_data_type_ has not been set." << std::endl;
        throw std::logic_error("_selected_data_type_ has not been set.");
    }

    InitializeThreadsParameters();
    InitializeFitParameters();
    InitializeDataSamples();
    InitializeFitter();

    _is_initialized_ = true;
    LogWarning << "ND280 Fitter has been initialized." << std::endl;

}


void ND280Fitter::WritePrefitData(){

    LogInfo << "Writing prefit data..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);


    /////////////////////////////
    // Samples
    /////////////////////////////
    LogDebug << "Writing prefit samples plots..." << std::endl;
    std::vector<Color_t> colorCells = {kGreen-3, kTeal+3, kAzure+7, kCyan-2, kBlue-7, kBlue+2, kOrange+1, kOrange+9, kRed+2, kPink+9};

    GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit")->cd();

    std::map<std::string, TH1D*> TH1D_handler;
    int iColor = 0;
    GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit/samples")->cd();
    for(const auto& anaSample : _samplesList_){

        std::map<std::string, TH1D*> tempHistMap;

        LogDebug << "Preparing prefit plots: " << anaSample->GetName() << std::endl;
        std::string tempHistName;

        // Build the histograms binning
        bool isSimpleBinning = true; // TODO: check if a simple binning can be applied
        std::vector<double> D1binning;
        auto bins = anaSample->GetBinEdges();
        for(const auto& bin : bins){
            if(D1binning.empty()) D1binning.emplace_back(bin.D1low);
            if(not GenericToolbox::doesElementIsInVector(bin.D1high, D1binning)){
                D1binning.emplace_back(bin.D1high);
            }
        }
        std::sort(D1binning.begin(), D1binning.end());

        // Define Histograms
        tempHistName               = anaSample->GetName() + "/D1";
        tempHistMap[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
                                              D1binning.size() - 1, &D1binning[0]);

        // Histograms splitted by reaction
        std::vector<int> reactionsList;
        for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
            auto* anaEvent = anaSample->GetEvent(iEvent);
            if(not GenericToolbox::doesElementIsInVector(anaEvent->GetReaction(), reactionsList)){
                reactionsList.emplace_back(anaEvent->GetReaction());
            }
        }
        for(const auto& thisReaction : reactionsList){
            tempHistName = anaSample->GetName() + "/reactions/D1_reaction_"+std::to_string(thisReaction);
            tempHistMap[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
                                                  D1binning.size() - 1, &D1binning[0]);
        }


        // Fill the histograms
        for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
            auto* anaEvent = anaSample->GetEvent(iEvent);
            tempHistMap[anaSample->GetName() + "/D1"]->Fill(
                anaEvent->GetRecoD1()
                , anaEvent->GetEvWght()*anaSample->GetNorm()
            );
            tempHistMap[anaSample->GetName() + "/reactions/D1_reaction_"+std::to_string(anaEvent->GetReaction())]->Fill(
                anaEvent->GetRecoD1()
                , anaEvent->GetEvWght()
            );
        }

        std::function<void(TH1D*)> renormalizeHist = [anaSample](TH1D* hist_){
            // Rescale MC stat to Data
//            LogTrace << anaSample->GetName() << ": " << anaSample->GetNorm() << std::endl;
            hist_->Scale(anaSample->GetNorm());

            // Get Number of counts per 100 MeV
            for(int iBin = 0 ; iBin < hist_->GetNbinsX() ; iBin++){
                hist_->SetBinContent(iBin,hist_->GetBinContent(iBin)/hist_->GetBinWidth(iBin)*100.);
                hist_->SetBinError(iBin,hist_->GetBinError(iBin)/hist_->GetBinWidth(iBin)*100.);
            }
        };

        std::function<void(TH1D*)> makeupHist = [colorCells, iColor](TH1D* hist_){
            hist_->SetLineColor(colorCells[iColor]);
            hist_->SetLineWidth(2);
            hist_->SetMarkerColor(colorCells[iColor]);
            if(GenericToolbox::doesStringContainsSubstring(hist_->GetName(), "D1")){
                hist_->GetXaxis()->SetTitle("D1 Reco");
            }
            hist_->GetYaxis()->SetTitle("Counts/(100 MeV)");
            hist_->GetYaxis()->SetRangeUser(0, hist_->GetMaximum()*1.2);
        };


        for(auto& histPair : tempHistMap){
            renormalizeHist(histPair.second);
            makeupHist(histPair.second);
        }

        GenericToolbox::appendToMap(TH1D_handler, tempHistMap);

        // Next Loop
        iColor++;
//        samplesDir->cd();
    }


    // Saving Histograms
    for(auto& hist : TH1D_handler){
        // Looking for a subfolder in the hist name
        auto nameSlices = GenericToolbox::splitString(hist.second->GetName(), "/");
        std::string histName = nameSlices[nameSlices.size()-1];
        std::string histSubFolder = GenericToolbox::joinVectorString(nameSlices, "/", 0, -1);
        GenericToolbox::mkdirTFile(_outputTDirectory_, Form("prefit/samples/%s",histSubFolder.c_str()))->cd();

        // Writing histo
        hist.second->Write(histName.c_str());
    }

    // Building canvas
    std::map<int, std::pair<std::string, Color_t>> reactionNamesAndColors = {
        {-1, {"no truth", kBlack}},
        {0, {"CCQE", kOrange+1}},
        {1, {"RES", kGreen-3}},
        {2, {"DIS", kTeal+3}},
        {3, {"COH", kAzure+7}},
        {4, {"NC", kCyan-2}},
        {5, {"CC-#bar{#nu}_{#mu}", kBlue-7}},
        {6, {"CC-#nu_{e}, CC-#bar{#nu}_{e}", kBlue+2}},
        {7, {"out FV", kPink+9}},
        {9, {"2p2h", kRed+2}},
        {777, {"sand #mu", kGray}},
        {999, {"other", kGray+2}},
    };
    std::map<std::string, TCanvas*> canvasHandler;
    std::string canvasD1ReactionName = "D1_reactions";
    canvasHandler[canvasD1ReactionName] = new TCanvas(canvasD1ReactionName.c_str(), canvasD1ReactionName.c_str(), 1200, 700);
    canvasHandler[canvasD1ReactionName]->Divide(3,2);
    int iSlot = 1;
    int iCanvas = 1;
    for(const auto& anaSample : _samplesList_){
        // select the canvas slot
        if(iSlot > 6){
            iSlot = 1;
            iCanvas++;
            canvasD1ReactionName = "D1_reaction_" + std::to_string(iCanvas);
            canvasHandler[canvasD1ReactionName] = new TCanvas(canvasD1ReactionName.c_str(), canvasD1ReactionName.c_str(), 1200, 700);
            canvasHandler[canvasD1ReactionName]->Divide(3,2);
            canvasHandler[canvasD1ReactionName]->cd();
        }
        canvasHandler[canvasD1ReactionName]->cd(iSlot);

        // Select which histograms which be displayed
        std::vector<TH1D*> histToPlotList;
        std::vector<int> reactionCodes;
        for(auto& hist: TH1D_handler){
            if(not GenericToolbox::doesStringContainsSubstring(hist.first, anaSample->GetName())) continue; // select the sample
            std::string histName = GenericToolbox::splitString(hist.first, "/").back();
            auto histNameSlices = GenericToolbox::splitString(histName, "_reaction_");

            // he's a reaction histogram ?
            if(histNameSlices.size() < 2) continue; // if not, skip
            histToPlotList.emplace_back(hist.second);
            reactionCodes.emplace_back(stoi(histNameSlices[1]));
        }
        if(histToPlotList.empty()) continue;

        // Stacking histograms
        TH1D* tempHist = nullptr;
        for( int iHist = int(histToPlotList.size())-1 ; iHist >= 0 ; iHist-- ){
            if(tempHist != nullptr) histToPlotList[iHist]->Add(tempHist);
            tempHist = histToPlotList[iHist];
        }
        double lastBinLowEdge = histToPlotList[0]->GetXaxis()->GetBinLowEdge(histToPlotList[0]->GetXaxis()->GetNbins());
        histToPlotList[0]->GetXaxis()->SetRange(0,lastBinLowEdge);
        histToPlotList[0]->GetXaxis()->SetRangeUser(0,2000);
        double maxYplot = histToPlotList[0]->GetMaximum()*1.2;
        histToPlotList[0]->GetYaxis()->SetRangeUser(0, maxYplot);

        // Colors
        for( int iHist = 0 ; iHist < histToPlotList.size() ; iHist++ ){
            histToPlotList[iHist]->SetTitle(reactionNamesAndColors[reactionCodes[iHist]].first.c_str()); // set the name for the legend
            histToPlotList[iHist]->SetLineColor(reactionNamesAndColors[reactionCodes[iHist]].second);
            histToPlotList[iHist]->SetMarkerColor(reactionNamesAndColors[reactionCodes[iHist]].second);
            histToPlotList[iHist]->SetFillColor(reactionNamesAndColors[reactionCodes[iHist]].second);
            histToPlotList[iHist]->SetLineWidth(2);
            if(iHist == 0) histToPlotList[iHist]->Draw("HIST");
            else histToPlotList[iHist]->Draw("HISTSAME");
        }
        gPad->BuildLegend(0.6,0.5,0.9,0.9);
        histToPlotList[0]->SetTitle(anaSample->GetName().c_str());
        gPad->SetGridx();
        gPad->SetGridy();
        iSlot++;
    }

    _outputTDirectory_->cd("prefit/samples");
    for(auto& canvas : canvasHandler){
        canvas.second->Write();
    }


    /////////////////////////////
    // Systematics
    /////////////////////////////
    LogDebug << "Writing prefit systematics data..." << std::endl;

    TH1D* histPrefitParameters =
        new TH1D("histPrefitParameters","histPrefitParameters",
                 _nb_fit_parameters_,0, _nb_fit_parameters_);
    TVectorD vecPrefitParameters        (_nb_fit_parameters_);
    TVectorD vecPrefitDecompParameters  (_nb_fit_parameters_);
    TVectorD vecPrefitStartParameters   (_nb_fit_parameters_, _parameterPriorValues_.data());

    int parGlobalIndex = 1;
    for(const auto & anaFitParameters : _fitParametersGroupList_){

        GenericToolbox::mkdirTFile(_outputTDirectory_, Form("prefit/systematics/%s", anaFitParameters->GetName().c_str()))->cd();

        auto* covMatrix = anaFitParameters->GetCovMat();
        auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
        auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s", anaFitParameters->GetName().c_str()));
        auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s", anaFitParameters->GetName().c_str()));

        TH1D* histPrefitParametersCategory = new TH1D("histPrefitParameters","histPrefitParameters",
                                                      _nb_fit_parameters_,0, _nb_fit_parameters_);
        TH1D* histPrefitParametersError = new TH1D("histPrefitParametersError","histPrefitParametersError",
                                                      _nb_fit_parameters_,0, _nb_fit_parameters_);

        for(int iPar = 0 ; iPar < anaFitParameters->GetNpar() ; ++iPar){

            // Parameter Prior Value
            histPrefitParameters->SetBinContent(parGlobalIndex, anaFitParameters->GetParPrior(iPar));
            histPrefitParametersCategory->SetBinContent(iPar+1, anaFitParameters->GetParPrior(iPar));

            // Parameter Prior Error
            if(anaFitParameters->HasCovMat()){
                histPrefitParameters->SetBinError(parGlobalIndex, std::sqrt((*anaFitParameters->GetCovMat())[iPar][iPar]));
                histPrefitParametersCategory->SetBinError(iPar+1, std::sqrt((*anaFitParameters->GetCovMat())[iPar][iPar]));
                histPrefitParametersError->SetBinContent(iPar+1, std::sqrt((*anaFitParameters->GetCovMat())[iPar][iPar]));
            }
            else{
                histPrefitParameters->SetBinError(parGlobalIndex, 0);
                histPrefitParametersCategory->SetBinError(iPar+1, 0);
            }

            // Fill Vectors
            vecPrefitParameters[parGlobalIndex -1] = anaFitParameters->GetParOriginal(iPar);
            vecPrefitDecompParameters[parGlobalIndex -1] = anaFitParameters->GetParPrior(iPar);

            // Labels
            std::vector<std::string> vecBuffer;
            anaFitParameters->GetParNames(vecBuffer);
            histPrefitParameters        ->GetXaxis()->SetBinLabel(parGlobalIndex, vecBuffer[iPar].c_str());
            histPrefitParametersCategory->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            histPrefitParametersError   ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            covMatrixTH2D               ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            covMatrixTH2D               ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            corMatrixTH2D               ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            corMatrixTH2D               ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());

            // Next index
            parGlobalIndex++;
        }

        histPrefitParametersCategory->Write();
        histPrefitParametersError->Write();
        covMatrix->Write("CovarianceMatrix");
        corMatrix->Write("CorrelationMatrix");
        covMatrixTH2D->Write("CovarianceMatrix_TH2D");
        corMatrixTH2D->Write("CorrelationMatrix_TH2D");

    }

    _outputTDirectory_->cd("prefit/systematics");

    histPrefitParameters->Write("histPrefitParameters");
    vecPrefitParameters.Write("vecPrefitParameters");
    vecPrefitDecompParameters.Write("vecPrefitDecompParameters");
    vecPrefitStartParameters.Write("vecPrefitStartParameters");

    TMatrixD* covTMatrixD = GeneratePriorCovarianceMatrix();
    covTMatrixD->Write("PriorCovarianceMatrix_TMatrixD");
    TMatrixD* corTMatrixD = GenericToolbox::convertToCorrelationMatrix(covTMatrixD);
    corTMatrixD->Write("PriorCorrelationMatrix_TMatrixD");
    TH2D* tempTH2D = GenericToolbox::convertTMatrixDtoTH2D(covTMatrixD,"PriorCovarianceMatrix");
    tempTH2D->Write("PriorCovarianceMatrix_TH2D");
    tempTH2D = GenericToolbox::convertTMatrixDtoTH2D(corTMatrixD,"PriorCorrelationMatrix_TMatrixD");
    tempTH2D->Write("PriorCorrelationMatrix_TH2D");

    _outputTDirectory_->cd();

}
void ND280Fitter::MakeOneSigmaChecks(){

    LogInfo << "Making one sigma checks..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    TDirectory* oneSigmaDir;
    if(not _fitHasBeenDone_){
        TDirectory* fitDir = GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit");
        oneSigmaDir = GenericToolbox::mkdirTFile(fitDir, "oneSigma");
    }
    else{
        TDirectory* fitDir = GenericToolbox::mkdirTFile(_outputTDirectory_, "postfit");
        oneSigmaDir = GenericToolbox::mkdirTFile(fitDir, "oneSigma");
    }
    oneSigmaDir->cd();


    // Make sure all weights a applied
    LogDebug << "Re-computing weights..." << std::endl;
    this->PropagateSystematics();

    LogDebug << "Computing nominal MC histograms..." << std::endl;
    std::map<std::string, TH1D*> nominalHistMap;
    for(const auto& anaSample : _samplesList_){

        std::string tempHistName;

        // Build the histograms binning
        bool isSimpleBinning = true; // TODO: check if a simple binning can be applied
        std::vector<double> D1binning;
        auto bins = anaSample->GetBinEdges();
        for(const auto& bin : bins){
            if(D1binning.empty()) D1binning.emplace_back(bin.D1low);
            if(not GenericToolbox::doesElementIsInVector(bin.D1high, D1binning)){
                D1binning.emplace_back(bin.D1high);
            }
        }
        std::sort(D1binning.begin(), D1binning.end());

        // Define Histograms
        tempHistName                 = anaSample->GetName() + "/D1";
        nominalHistMap[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
                                             D1binning.size() - 1, &D1binning[0]);

        // Histograms splitted by reaction
        std::vector<int> reactionsList;
        for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
            auto* anaEvent = anaSample->GetEvent(iEvent);
            if(not GenericToolbox::doesElementIsInVector(anaEvent->GetReaction(), reactionsList)){
                reactionsList.emplace_back(anaEvent->GetReaction());
            }
        }
        for(const auto& thisReaction : reactionsList){
            tempHistName = anaSample->GetName() + "/reactions/D1_reaction_"+std::to_string(thisReaction);
            nominalHistMap[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
                                                 D1binning.size() - 1, &D1binning[0]);
        }


        // Fill the histograms
        for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
            auto* anaEvent = anaSample->GetEvent(iEvent);
            nominalHistMap[anaSample->GetName() + "/D1"]->Fill(
                anaEvent->GetRecoD1()
                , anaEvent->GetEvWght()
//                    *anaSample->GetNorm()
            );
        }

    }


    LogDebug << "Performing individual one sigma variations..." << std::endl;
    TMatrixD* covarianceMatrix = this->GeneratePriorCovarianceMatrix();
    std::vector<std::map<std::string, TH1D*>> oneSigmaHistMap(_minimizer_->NDim());
    for( int iSyst = 0 ; iSyst < _minimizer_->NDim() ; iSyst++ ){

        // If this parameter is fixed, then it won't be writen
        if(_fixParameterStatusList_[iSyst]){
            LogTrace << "Skipping fixed parameter: " << iSyst << std::endl;
            continue;
        }

        LogDebug << "Running +1 sigma on: " << _parameter_names_[iSyst] << std::endl;

        // Create associated subdirectory
        auto* currentDir = GenericToolbox::mkdirTFile( oneSigmaDir, _parameter_names_[iSyst].c_str() );
        currentDir->cd();

        // Put the parameter at 1 sigma
        int systCandidate = 0;
        double syst_val = -1;
        double syst_shift = -1;
        for(int jCat = 0; jCat < _fitParametersGroupList_.size(); ++jCat){
            for(int jPar = 0 ; jPar < _fitParametersGroupList_[jCat]->GetNpar() ; jPar++){
                if(systCandidate == iSyst){
                    syst_val = _parameterPriorValues_[iSyst];
                    _newParametersList_[jCat][jPar] = _parameterPriorValues_[iSyst] + TMath::Sqrt((*covarianceMatrix)[iSyst][iSyst]);
                    syst_shift = TMath::Sqrt((*covarianceMatrix)[iSyst][iSyst]);
//                    LogTrace << GET_VAR_NAME_VALUE(iSyst) << std::endl;
//                    LogTrace << GET_VAR_NAME_VALUE(syst_val) << std::endl;
//                    LogTrace << GET_VAR_NAME_VALUE(syst_shift) << std::endl;
                }
                systCandidate++;
            }
        }

        // Propagate systematics
        this->PropagateSystematics();

        std::vector<Color_t> colorWheel = {
            kGreen-3,
            kTeal+3,
            kAzure+7,
            kCyan-2,
            kBlue-7,
            kBlue+2
        };
        int colorTick = 0;

        // Generate plots
        for( const auto& anaSample : _samplesList_){

            std::string tempHistName = anaSample->GetName() + "/D1";

            oneSigmaHistMap[iSyst][tempHistName] = (TH1D*) nominalHistMap[tempHistName]->Clone();
            oneSigmaHistMap[iSyst][tempHistName]->SetName   (tempHistName.c_str());
            oneSigmaHistMap[iSyst][tempHistName]->SetTitle  (tempHistName.c_str());
            oneSigmaHistMap[iSyst][tempHistName]->GetYaxis()->SetTitle("Deviation to Nominal (%)");
            oneSigmaHistMap[iSyst][tempHistName]->GetXaxis()->SetTitle("D1");
            oneSigmaHistMap[iSyst][tempHistName]->SetLineWidth(2);
            oneSigmaHistMap[iSyst][tempHistName]->SetLineColor(colorWheel[colorTick%colorWheel.size()]);
            oneSigmaHistMap[iSyst][tempHistName]->SetMarkerColor(colorWheel[colorTick%colorWheel.size()]);
            colorTick++;

            // Reset all bins (0, 1 to N, N+1)
            for(int iBin = 0 ; iBin <= oneSigmaHistMap[iSyst][tempHistName]->GetNbinsX() + 1 ; iBin++){
                oneSigmaHistMap[iSyst][tempHistName]->SetBinContent(iBin, 0);
            }

            // Fill the histograms
            for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
                auto* anaEvent = anaSample->GetEvent(iEvent);
                oneSigmaHistMap[iSyst][tempHistName]->Fill(
                    anaEvent->GetRecoD1()
                    , anaEvent->GetEvWght()
                );
//                if(anaEvent->GetEvWght() != 1) LogAlert << GET_VAR_NAME_VALUE(anaEvent->GetEvWght()) << std::endl;
            }

            bool isUnaffected = true;
            for(int iBin = 0 ; iBin <= oneSigmaHistMap[iSyst][tempHistName]->GetNbinsX() + 1 ; iBin++){
                oneSigmaHistMap[iSyst][tempHistName]->SetBinContent(
                    iBin,(oneSigmaHistMap[iSyst][tempHistName]->GetBinContent(iBin)/nominalHistMap[tempHistName]->GetBinContent(iBin) - 1)*100.
                    );
                oneSigmaHistMap[iSyst][tempHistName]->SetBinError(
                    iBin,
                    oneSigmaHistMap[iSyst][tempHistName]->GetBinError(iBin)/nominalHistMap[tempHistName]->GetBinContent(iBin)*100.
                );
                if(TMath::Abs(oneSigmaHistMap[iSyst][tempHistName]->GetBinContent(iBin)) > 0.1){
                    isUnaffected = false;
                }
            }
            if(isUnaffected){
                LogWarning << anaSample->GetName() << " is unaffected by " << _parameter_names_[iSyst] << std::endl;
            }

            // Y axis
            double Ymin = 0;
            double Ymax = 0;
            for(int iBin = 0 ; iBin <= oneSigmaHistMap[iSyst][tempHistName]->GetNbinsX() + 1 ; iBin++){
                double val = oneSigmaHistMap[iSyst][tempHistName]->GetBinContent(iBin);
                double error = oneSigmaHistMap[iSyst][tempHistName]->GetBinError(iBin);
                if(val + error > Ymax){
                    Ymax = val + error;
                }
                if(val - error < Ymin){
                    Ymin = val - error;
                }
            }

            // Add 20% margin
            Ymin += Ymin*0.2;
            Ymax += Ymax*0.2;

            // Force showing Y=0
            if(Ymin > -0.2) Ymin = -0.2;
            if(Ymax < 0.2) Ymax = 0.2;

            oneSigmaHistMap[iSyst][tempHistName]->GetYaxis()->SetRangeUser(Ymin, Ymax);
            oneSigmaHistMap[iSyst][tempHistName]->GetXaxis()->SetRangeUser(0, 2000); //! TODO: COMMENT THIS LINE IN THE FUTURE

        }
        GenericToolbox::mkdirTFile(currentDir,"Histograms")->cd();
        for(auto& histPair : oneSigmaHistMap[iSyst]){
            histPair.second->Write();
        }

        // Building canvas
        currentDir->cd();
        int iSlot = 1;
        int iCanvas = 1;
        std::map<std::string, TCanvas*> canvasHandler;
        std::string canvasBaseD1Name = _parameter_names_[iSyst] + "_D1_";
        std::string canvasD1Name    = canvasBaseD1Name + std::to_string(iCanvas);
        canvasHandler[canvasD1Name] = new TCanvas(canvasD1Name.c_str(), canvasD1Name.c_str(), 1200, 700);
        canvasHandler[canvasD1Name]->Divide(3,2);
        for(const auto& anaSample : _samplesList_){
            // select the canvas slot
            if(iSlot > 6){
                iSlot = 1;
                iCanvas++;
                canvasD1Name                = canvasBaseD1Name + std::to_string(iCanvas);
                canvasHandler[canvasD1Name] = new TCanvas(canvasD1Name.c_str(), canvasD1Name.c_str(), 1200, 700);
                canvasHandler[canvasD1Name]->Divide(3,2);
                canvasHandler[canvasD1Name]->cd();
            }
            canvasHandler[canvasD1Name]->cd(iSlot);

            std::string tempHistName = anaSample->GetName() + "/D1";
            oneSigmaHistMap[iSyst][tempHistName]->Draw("E");

            auto* nominalLine = new TLine(
                oneSigmaHistMap[iSyst][tempHistName]->GetXaxis()->GetXmin(), 0,
                oneSigmaHistMap[iSyst][tempHistName]->GetXaxis()->GetXmax(), 0
                );
            nominalLine->SetLineStyle(2);
            nominalLine->SetLineWidth(3);
            nominalLine->SetLineColor(kGray);
            nominalLine->Draw("SAME");

            gPad->SetGridx();
            gPad->SetGridy();
            iSlot++;
        }
        for(const auto& canvas: canvasHandler){
            canvas.second->Write((canvas.first + "_TCanvas").c_str());
            delete canvas.second;
        }

        // Get back to 0 sigmas
        systCandidate = 0;
        for(int jCat = 0; jCat < _fitParametersGroupList_.size(); ++jCat){
            for(int jPar = 0 ; jPar < _fitParametersGroupList_[jCat]->GetNpar() ; jPar++){
                if(systCandidate == iSyst){
                    _newParametersList_[jCat][jPar] = _parameterPriorValues_[iSyst];
                }
                systCandidate++;
            }
        }

        LogDebug << std::endl;

    }

    _outputTDirectory_->cd();

    LogInfo << "Make one sigma check just ended." << std::endl;

}
bool ND280Fitter::Fit(){

    LogAlert << "Starting to fit." << std::endl;
    PassIfInitialized(__METHOD_NAME__);
    _fitHasBeenDone_ = true;

    LogInfo << "Calling Minimize, running: " << _minimization_settings_.algorithm << std::endl;

    // Run the actual fitter:
    _fit_has_converged_ = _minimizer_->Minimize();
    _fit_has_converged_ = true;

    if(not _fit_has_converged_){
        LogError << "Fit did not converge while running " << _minimization_settings_.algorithm
                 << std::endl;
        LogError << "Failed with status code: " << _minimizer_->Status() << std::endl;
    }
    else{
        LogInfo << "Fit converged." << std::endl
                << "Status code: " << _minimizer_->Status() << std::endl;

        LogInfo << "Calling HESSE." << std::endl;
        _fit_has_converged_ = _minimizer_->Hesse();
    }

    if(not _fit_has_converged_){
        LogError  << "Hesse did not converge." << std::endl;
        LogError  << "Failed with status code: " << _minimizer_->Status() << std::endl;
    }
    else{
        LogInfo << "Hesse converged." << std::endl
                << "Status code: " << _minimizer_->Status() << std::endl;
    }

    LogWarning << "Fit has ended." << std::endl;
    return _fit_has_converged_;

}
void ND280Fitter::WritePostFitData(){

    LogWarning << "Writing postfit data..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    if(not _fitHasBeenDone_){
        LogFatal << "Can't write post fit data: fit has not been called previously" << std::endl;
        throw std::logic_error("Can't write post fit data: fit has not been called previously");
    }

    GenericToolbox::mkdirTFile(_outputTDirectory_, "fit")->cd();

    LogInfo << "Saving Chi2 History..." << std::endl;

    GenericToolbox::convertTVectorDtoTH1D(_chi2StatHistory_, "_chi2StatHistory_")->Write("chi2StatHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2PullsHistory_, "_chi2PullsHistory_")->Write("chi2PullsHistory");
    GenericToolbox::convertTVectorDtoTH1D(_chi2RegHistory_, "_chi2RegHistory_")->Write("chi2RegHistory");

    auto chi2TotHistory = _chi2StatHistory_;
    for(size_t it = 0 ; it < chi2TotHistory.size() ; it++){ chi2TotHistory[it] += _chi2PullsHistory_[it] + _chi2RegHistory_[it]; }
    GenericToolbox::convertTVectorDtoTH1D(chi2TotHistory, "chi2TotHistory")->Write("chi2TotHistory");


    GenericToolbox::mkdirTFile(_outputTDirectory_, "postfit")->cd();

    LogInfo << "Writing postfit parameters..." << std::endl;

    const int nfree = _minimizer_->NFree();

    if(_minimizer_->X() != nullptr){

        double covarianceMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
        _minimizer_->GetCovMatrix(covarianceMatrixArray);
        TMatrixDSym covarianceMatrix(_minimizer_->NDim(), covarianceMatrixArray);

        std::vector<double> parameterValueList(_minimizer_->X(),      _minimizer_->X()      + _minimizer_->NDim());
        std::vector<double> parameterErrorList(_minimizer_->Errors(), _minimizer_->Errors() + _minimizer_->NDim());

        int paramaterOffset = 0;
        for(const auto& anaFitParameters : _fitParametersGroupList_){

            GenericToolbox::mkdirTFile(_outputTDirectory_, Form("postfit/systematics/%s", anaFitParameters->GetName().c_str()))->cd();

            auto* covMatrix = (TMatrixD*) anaFitParameters->GetCovMat()->Clone(); // just for the tempate template
            for(int iComp = 0 ; iComp < anaFitParameters->GetNpar() ; iComp++){
                for(int jComp = 0 ; jComp < anaFitParameters->GetNpar() ; jComp++){
                    (*covMatrix)[iComp][jComp] = covarianceMatrix[paramaterOffset+iComp][paramaterOffset+jComp];
                } // jComp
            } // iComp
            auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
            auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s", anaFitParameters->GetName().c_str()));
            auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s", anaFitParameters->GetName().c_str()));

            TH1D* histPostfitParametersCategory = new TH1D("histPostfitParameters","histPostfitParameters",
                                                          _nb_fit_parameters_,0, _nb_fit_parameters_);
            TH1D* histPostfitParametersError = new TH1D("histPostfitParametersError","histPostfitParametersError",
                                                       _nb_fit_parameters_,0, _nb_fit_parameters_);

            for(int iPar = 0 ; iPar < anaFitParameters->GetNpar() ; ++iPar){

                // Parameter Prior Value
                histPostfitParametersCategory->SetBinContent(iPar+1, anaFitParameters->GetParPrior(iPar));

                // Parameter Prior Error
                if(anaFitParameters->HasCovMat()){
                    histPostfitParametersCategory->SetBinError(iPar+1, std::sqrt((*anaFitParameters->GetCovMat())[iPar][iPar]));
                    histPostfitParametersError->SetBinContent(iPar+1, std::sqrt((*anaFitParameters->GetCovMat())[iPar][iPar]));
                }
                else{
                    histPostfitParametersCategory->SetBinError(iPar+1, 0);
                }

                // Fill Vectors
//                vecPostfitParameters[parGlobalIndex -1] = anaFitParameters->GetParOriginal(iPar);
//                vecPostfitDecompParameters[parGlobalIndex -1] = anaFitParameters->GetParPrior(iPar);

                // Labels
                std::vector<std::string> vecBuffer;
                anaFitParameters->GetParNames(vecBuffer);
                histPostfitParametersCategory->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                histPostfitParametersError   ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                covMatrixTH2D                ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                covMatrixTH2D                ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                corMatrixTH2D                ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                corMatrixTH2D                ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
            }

            histPostfitParametersCategory->Write();
            histPostfitParametersError->Write();
            covMatrix->Write("CovarianceMatrix");
            corMatrix->Write("CorrelationMatrix");
            covMatrixTH2D->Write("CovarianceMatrix_TH2D");
            corMatrixTH2D->Write("CorrelationMatrix_TH2D");

            paramaterOffset += anaFitParameters->GetNpar();
        } // fit Parameters

    }



//    cwd->cd();
//    cov_matrix.Write("res_cov_matrix");
//    cor_matrix.Write("res_cor_matrix");
//    postfit_param.Write("res_vector");
//    postfit_globalcc.Write("res_globalcc");
//
//    SaveResults(res_pars, err_pars);
//    SaveEventHist(_nbFitCalls_, true);
//
//    if(_save_event_tree_) SaveEventTree(res_pars);

    _outputTDirectory_->cd();

}


void ND280Fitter::InitializeThreadsParameters(){

    if(GlobalVariables::getNbThreads() > 1){

        LogWarning << "Initializing threads workers..." << std::endl;

        _histThreadHandlers_.resize(GlobalVariables::getNbThreads());
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++){
            _histThreadHandlers_[iThread].resize(_samplesList_.size());
            for(size_t iSample = 0 ; iSample < _samplesList_.size() ; iSample++){
                _histThreadHandlers_[iThread][iSample].resize(4, nullptr);
                if(_samplesList_[iSample]->GetPredHisto() != nullptr){
                    _histThreadHandlers_[iThread][iSample][0] = (TH1D*)_samplesList_[iSample]->GetPredHisto()->Clone();
                }
                if(_samplesList_[iSample]->GetMCHisto() != nullptr){
                    _histThreadHandlers_[iThread][iSample][1] = (TH1D*)_samplesList_[iSample]->GetMCHisto()->Clone();
                }
                if(_samplesList_[iSample]->GetMCTruthHisto() != nullptr){
                    _histThreadHandlers_[iThread][iSample][2] = (TH1D*)_samplesList_[iSample]->GetMCTruthHisto()->Clone();
                }
                if(_samplesList_[iSample]->GetSignalHisto() != nullptr){
                    _histThreadHandlers_[iThread][iSample][3] = (TH1D*)_samplesList_[iSample]->GetSignalHisto()->Clone();
                }
            }
        }

        std::function<void(int)> asyncLoop = [this](int iThread_){
            while(not _stopThreads_){
                // Pending state loop
//                std::this_thread::sleep_for(std::chrono::microseconds(33)); // 30,000 fps cap

                if(_triggerReweightThreads_.at(iThread_)){
                    this->ReWeightEvents(iThread_);
                    _triggerReweightThreads_.at(iThread_) = false; // toggle off the trigger
                }

                if(_triggerReFillMcHistogramsThreads_.at(iThread_)){
                    this->ReFillSampleMcHistograms(iThread_);
                    _triggerReFillMcHistogramsThreads_.at(iThread_) = false; // toggle off the trigger
                }

                // Add other jobs there
            }
            GlobalVariables::getThreadMutex().lock();
            LogDebug << "Thread " << iThread_ << " will end now." << std::endl;
            GlobalVariables::getThreadMutex().unlock();
        };

        // Parallel loops start now:
        _triggerReweightThreads_.clear();
        _triggerReFillMcHistogramsThreads_.clear();
        _stopThreads_ = false;
        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++ ){
            _triggerReweightThreads_.push_back(false); // no emplace back for bools
            _triggerReFillMcHistogramsThreads_.push_back(false);
            _asyncFitThreads_.emplace_back(
                std::async( std::launch::async, std::bind(asyncLoop, iThread) )
            );
        }

    }

}
void ND280Fitter::InitializeFitParameters(){

    LogWarning << "Initializing the fit parameters..." << std::endl;

    _durationReweightParameters_.resize(_fitParametersGroupList_.size());

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(size_t iGroup = 0 ; iGroup < _fitParametersGroupList_.size() ; iGroup++) {

        _durationReweightParameters_[iGroup].resize(GlobalVariables::getNbThreads());

        // _nb_fit_parameters_ is the number of total fit paramters:
        _nb_fit_parameters_ += _fitParametersGroupList_[iGroup]->GetNpar();

        // Get names of all the different parameters (for all parameter types) and store them in par_names:
        std::vector<std::string> ana_par_names;
        _fitParametersGroupList_[iGroup]->GetParNames(ana_par_names);
        _parameter_names_.insert(_parameter_names_.end(), ana_par_names.begin(),ana_par_names.end());

        // Get the priors for this parameter type (should be 1 unless decomp has been set to true in the .json config file) and store them in vec1:
        std::vector<double> parListBuffer;
        _fitParametersGroupList_[iGroup]->GetParPriors(parListBuffer);

        // If rng_template has been set to true in the .json config file, the template parameters will be randomized (around 1) according to a gaussian distribution:
        if(_fitParametersGroupList_[iGroup]->DoRNGstart())
        {
            LogInfo << "Randomizing start point for " << _fitParametersGroupList_[iGroup]->GetName() << std::endl;
            for(auto& p : parListBuffer)
                p += (p * _PRNG_->Gaus(0.0, 0.1));
        }

        // Store the prefit values (for all parameter types) in par_prefit:
        _parameterPriorValues_.insert(_parameterPriorValues_.end(), parListBuffer.begin(),
                                         parListBuffer.end());
        _newParametersList_.emplace_back(parListBuffer);

        // Store the pars_step values (for all parameter types) and store them in _parameterSteps_:
        _fitParametersGroupList_[iGroup]->GetParSteps(parListBuffer);
        _parameterSteps_.insert(_parameterSteps_.end(),
                                parListBuffer.begin(),
                                parListBuffer.end());

        // Store the lower and upper limits for the fit parameters (for all parameter types) in _parameter_low_edges_ and _parameter_high_edges_:
        std::vector<double> anaParLowEdges, anaParHighEdges;
        _fitParametersGroupList_[iGroup]->GetParLimits(anaParLowEdges, anaParHighEdges);
        _parameter_low_edges_.insert( _parameter_low_edges_.end(), anaParLowEdges.begin(), anaParLowEdges.end() );
        _parameter_high_edges_.insert( _parameter_high_edges_.end(), anaParHighEdges.begin(), anaParHighEdges.end() );

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in _fixParameterStatusList_:
        std::vector<bool> anaParFixedList;
        _fitParametersGroupList_[iGroup]->GetParFixed(anaParFixedList);
        _fixParameterStatusList_.insert(_fixParameterStatusList_.end(), anaParFixedList.begin(), anaParFixedList.end());
    }

    // Nothing to fit with zero fit parameters:
    if(_nb_fit_parameters_ == 0){
        LogError << "No fit parameters were defined." << std::endl;
        return;
    }

    // Prior Re-Weighting
    LogInfo << "Applying prior systematics..." << std::endl;
    this->PropagateSystematics();

}
void ND280Fitter::InitializeDataSamples(){

    if(_selected_data_type_ == kReset){
        LogFatal << "In " << __METHOD_NAME__ << std::endl;
        LogFatal << "No valid _selected_data_type_ provided." << std::endl;
        throw std::logic_error("No valid _selected_data_type_ provided.");
    }
    else if(_selected_data_type_ == kExternal){
        GenerateToyData(0, _apply_statistical_fluctuations_on_samples_);
    }
    else{
        for(const auto& anaSample : _samplesList_){
            anaSample->FillEventHist(
                _selected_data_type_,
                _apply_statistical_fluctuations_on_samples_
            );
        }
    }
}
void ND280Fitter::InitializeFitter(){

    // Print information about the minimizer settings specified in the .json config file:
    LogWarning << "Initializing the fitter..." << std::endl;

    LogInfo << "Minimizer settings..." << std::endl
            << "Minimizer: " << _minimization_settings_.minimizer << std::endl
            << "Algorithm: " << _minimization_settings_.algorithm << std::endl
            << "Likelihood: " << _minimization_settings_.likelihood << std::endl
            << "Strategy : " << _minimization_settings_.strategy << std::endl
            << "Print Lvl: " << _minimization_settings_.print_level << std::endl
            << "Tolerance: " << _minimization_settings_.tolerance << std::endl
            << "Max Iterations: " << _minimization_settings_.max_iter << std::endl
            << "Max Fcn Calls : " << _minimization_settings_.max_fcn << std::endl;

    // Create ROOT minimizer of given minimizerType and algoType:
    _minimizer_ = ROOT::Math::Factory::CreateMinimizer(_minimization_settings_.minimizer.c_str(),
                                                       _minimization_settings_.algorithm.c_str());

    // The ROOT Functor class is used to wrap multi-dimensional function objects, in this case the ND280Fitter::EvalFit function calculates and returns chi2_stat + chi2_sys + chi2_reg in each iteration of the fitter:
    _functor_ = new ROOT::Math::Functor(this, &ND280Fitter::EvalFit, _nb_fit_parameters_);

    _minimizer_->SetFunction(*_functor_);
    _minimizer_->SetStrategy(_minimization_settings_.strategy);
    _minimizer_->SetPrintLevel(_minimization_settings_.print_level);
    _minimizer_->SetTolerance(_minimization_settings_.tolerance);
    _minimizer_->SetMaxIterations(_minimization_settings_.max_iter);
    _minimizer_->SetMaxFunctionCalls(_minimization_settings_.max_fcn);

    for(int iPar = 0; iPar < _nb_fit_parameters_; iPar++){
        _minimizer_->SetVariable(
            iPar,
            _parameter_names_[iPar],
            _parameterPriorValues_[iPar],
            _parameterSteps_[iPar]
        );

        if(_fixParameterStatusList_[iPar]){
            _minimizer_->FixVariable(iPar);
        }
    }

    LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
            << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
            << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
            << std::endl;

}

void ND280Fitter::PassIfInitialized(const std::string& method_name_) const{
    if(not _is_initialized_){
        LogFatal << "Can't do " << method_name_ << " while not initialized." << std::endl;
        throw std::logic_error("Can't do " + method_name_ + " while not initialized.");
    }
}
void ND280Fitter::PassIfNotInitialized(const std::string& method_name_) const{
    if(_is_initialized_){
        LogFatal << "Can't do " << method_name_ << " while already initialized." << std::endl;
        throw std::logic_error("Can't do " + method_name_ + " while already initialized.");
    }
}

void ND280Fitter::FixParameter(const std::string& par_name, const double& value)
{
    auto iter = std::find(_parameter_names_.begin(), _parameter_names_.end(), par_name);
    if(iter != _parameter_names_.end())
    {
        const int i = std::distance(_parameter_names_.begin(), iter);
        _minimizer_->SetVariable(i, _parameter_names_.at(i).c_str(), value, 0);
        _minimizer_->FixVariable(i);
        LogInfo << "Fixing parameter " << _parameter_names_.at(i) << " to value " << value
                  << std::endl;
    }
    else
    {
        LogError << "In function ND280Fitter::FixParameter()\n"
                   << "Parameter " << par_name << " not found!" << std::endl;
    }
}

// Initializes the fit by setting up the fit parameters and creating the ROOT minimizer:
void ND280Fitter::InitFitter(std::vector<AnaFitParameters*>& fitpara)
{
    // Vector of the different parameter types such as [template, flux, detector, cross section]:
    _fitParametersGroupList_ = fitpara;

    // Vectors holding the settings for the fit parameters (for all parameter types):
    std::vector<double> par_step, par_low, par_high;
    std::vector<bool> par_fixed;

    // ROOT random number interface (seed of 0 means that seed is automatically computed based on time):
    //TRandom3 rng(0);

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(std::size_t i = 0; i < _fitParametersGroupList_.size(); i++)
    {
        // m_npar is the number of total fit paramters:
        _nb_fit_parameters_ += _fitParametersGroupList_[i]->GetNpar();

        // Get names of all the different parameters (for all parameter types) and store them in par_names:
        std::vector<std::string> vec0;
        _fitParametersGroupList_[i]->GetParNames(vec0);
        _parameter_names_.insert(_parameter_names_.end(), vec0.begin(), vec0.end());

        // Get the priors for this parameter type (should be 1 unless decomp has been set to true in the .json config file) and store them in vec1:
        std::vector<double> vec1, vec2;
        _fitParametersGroupList_[i]->GetParPriors(vec1);

        // If rng_template has been set to true in the .json config file, the template parameters will be randomized (around 1) according to a gaussian distribution:
        if(_fitParametersGroupList_[i]->DoRNGstart())
        {
            LogInfo << "Randomizing start point for " << _fitParametersGroupList_[i]->GetName() << std::endl;
            for(auto& p : vec1)
                p += (p * _PRNG_->Gaus(0.0, 0.1));
        }

        // Store the prefit values (for all parameter types) in par_prefit:
        _parameterPriorValues_.insert(_parameterPriorValues_.end(), vec1.begin(), vec1.end());

        // Store the pars_step values (for all parameter types) and store them in _parameterSteps_:
        _fitParametersGroupList_[i]->GetParSteps(vec1);
        par_step.insert(par_step.end(), vec1.begin(), vec1.end());

        // Store the lower and upper limits for the fit parameters (for all parameter types) in _parameter_low_edges_ and _parameter_high_edges_:
        _fitParametersGroupList_[i]->GetParLimits(vec1, vec2);
        par_low.insert(par_low.end(), vec1.begin(), vec1.end());
        par_high.insert(par_high.end(), vec2.begin(), vec2.end());

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in _fixParameterStatusList_:
        std::vector<bool> vec3;
        _fitParametersGroupList_[i]->GetParFixed(vec3);
        par_fixed.insert(par_fixed.end(), vec3.begin(), vec3.end());
    }

    // Nothing to fit with zero fit parameters:
    if(_nb_fit_parameters_ == 0)
    {
        LogError << "No fit parameters were defined." << std::endl;
        return;
    }

    // Print information about the minimizer settings specified in the .json config file:
    std::cout << "===========================================" << std::endl;
    std::cout << "           Initializing fitter             " << std::endl;
    std::cout << "===========================================" << std::endl;

    LogInfo   << "Minimizer settings..." << std::endl
              << "Minimizer: "      << _minimization_settings_.minimizer    << std::endl
              << "Algorithm: "      << _minimization_settings_.algorithm    << std::endl
              << "Likelihood: "     << _minimization_settings_.likelihood   << std::endl
              << "Strategy : "      << _minimization_settings_.strategy     << std::endl
              << "Print Lvl: "      << _minimization_settings_.print_level  << std::endl
              << "Tolerance: "      << _minimization_settings_.tolerance    << std::endl
              << "Max Iterations: " << _minimization_settings_.max_iter     << std::endl
              << "Max Fcn Calls : " << _minimization_settings_.max_fcn      << std::endl;

    // Create ROOT minimizer of given minimizerType and algoType:
    _minimizer_ = ROOT::Math::Factory::CreateMinimizer(_minimization_settings_.minimizer.c_str(),
                                                       _minimization_settings_.algorithm.c_str());

    // The ROOT Functor class is used to wrap multi-dimensional function objects, in this case the ND280Fitter::EvalFit function calculates and returns chi2_stat + chi2_sys + chi2_reg in each iteration of the fitter:
    _functor_ = new ROOT::Math::Functor(this, &ND280Fitter::EvalFit, _nb_fit_parameters_);

    _minimizer_->SetFunction(*_functor_);
    _minimizer_->SetStrategy(_minimization_settings_.strategy);
    _minimizer_->SetPrintLevel(_minimization_settings_.print_level);
    _minimizer_->SetTolerance(_minimization_settings_.tolerance);
    _minimizer_->SetMaxIterations(_minimization_settings_.max_iter);
    _minimizer_->SetMaxFunctionCalls(_minimization_settings_.max_fcn);

    for(int i = 0; i < _nb_fit_parameters_; ++i)
    {
        _minimizer_->SetVariable(i, _parameter_names_[i], _parameterPriorValues_[i], par_step[i]);
        //_minimizer_->SetVariableLimits(i, _parameter_low_edges_[i], _parameter_high_edges_[i]);

        if(par_fixed[i] == true)
            _minimizer_->FixVariable(i);
    }

    LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
              << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
              << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
              << std::endl;

    TH1D h_prefit("hist_prefit_par_all", "hist_prefit_par_all", _nb_fit_parameters_, 0,
                  _nb_fit_parameters_);
    TVectorD v_prefit_original(_nb_fit_parameters_);
    TVectorD v_prefit_decomp(_nb_fit_parameters_);
    TVectorD v_prefit_start(_nb_fit_parameters_, _parameterPriorValues_.data());

    int num_par = 1;
    for(int i = 0; i < _fitParametersGroupList_.size(); ++i)
    {
        TMatrixDSym* cov_mat = _fitParametersGroupList_[i]->GetCovMat();
        for(int j = 0; j < _fitParametersGroupList_[i]->GetNpar(); ++j)
        {
            h_prefit.SetBinContent(num_par, _fitParametersGroupList_[i]->GetParPrior(j));
            if(_fitParametersGroupList_[i]->HasCovMat())
                h_prefit.SetBinError(num_par, std::sqrt((*cov_mat)[j][j]));
            else
                h_prefit.SetBinError(num_par, 0);

            v_prefit_original[num_par-1] = _fitParametersGroupList_[i]->GetParOriginal(j);
            v_prefit_decomp[num_par-1] = _fitParametersGroupList_[i]->GetParPrior(j);
            num_par++;
        }
    }

    _outputTDirectory_->cd();
    h_prefit.Write();
    v_prefit_original.Write("vec_prefit_original");
    v_prefit_decomp.Write("vec_prefit_decomp");
    v_prefit_start.Write("vec_prefit_start");
}
void ND280Fitter::GenerateToyData(int toy_type, bool stat_fluc) {
    int temp_seed = _PRNG_->GetSeed();
    double chi2_stat = 0.0;
    double chi2_syst = 0.0;
    std::vector<std::vector<double>> fitpar_throw;
    for(const auto& fitpar : _fitParametersGroupList_)
    {
        std::vector<double> toy_throw(fitpar->GetNpar(), 0.0);
        fitpar -> ThrowPar(toy_throw, temp_seed++);

        chi2_syst += fitpar -> GetChi2(toy_throw);
        fitpar_throw.emplace_back(toy_throw);
    }

    for(int s = 0; s < _samplesList_.size(); ++s)
    {
        const unsigned int N  = _samplesList_[s]->GetN();
        const std::string det = _samplesList_[s]->GetDetector();
#pragma omp parallel for num_threads(GlobalVariables::getNbThreads())
        for(unsigned int i = 0; i < N; ++i)
        {
            AnaEvent* ev = _samplesList_[s]->GetEvent(i);
            ev->ResetEvWght();
            for(int j = 0; j < _fitParametersGroupList_.size(); ++j)
                _fitParametersGroupList_[j]->ReWeight(ev, det, s, i, fitpar_throw[j]);
        }

        _samplesList_[s]->FillEventHist(kAsimov, stat_fluc);
        _samplesList_[s]->FillEventHist(kReset);
        chi2_stat += _samplesList_[s]->CalcChi2();
    }

    LogInfo << "Generated toy throw from parameters.\n"
              << "Initial Chi2 Syst: " << chi2_syst << std::endl
              << "Initial Chi2 Stat: " << chi2_stat << std::endl;

    SaveParams(fitpar_throw);
}


double ND280Fitter::EvalFit(const double* par){

    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(FIT_IT_TIME_POINT);

    _printFitState_ = false;

    _nbFitCalls_++;
    if(_nbFitCalls_ < 20 or _nbFitCalls_ % 100 == 0){
        _printFitState_ = true;
    }

    UpdateFitParameterValues(par);
    PropagateSystematics();
    ComputeChi2();

    ////////////////////////////////
    // Save current state
    _chi2StatHistory_.emplace_back(_chi2StatBuffer_);
    _chi2PullsHistory_.emplace_back(_chi2PullsBuffer_);
    _chi2RegHistory_.emplace_back(_chi2RegBuffer_);

    if(_nbFitCalls_ % _saveFitParamsFrequency_ == 0 and _save_fit_params_){
        SaveParams(_newParametersList_);
        SaveEventHist(_nbFitCalls_);
    }

    _chi2Buffer_ = _chi2StatBuffer_ + _chi2PullsBuffer_ + _chi2RegBuffer_;

    ////////////////////////////////
    // Print state
    if(_printFitState_){
        LogWarning << "Func Calls: " << _nbFitCalls_ << std::endl;
        LogWarning << "Chi2 total: " << _chi2Buffer_ << std::endl;
        LogWarning << "Chi2 stat : " << _chi2StatBuffer_ << std::endl
                   << "Chi2 syst : " << _chi2PullsBuffer_ << std::endl
                   << "Chi2 reg  : " << _chi2RegBuffer_ << std::endl;
    }

    _durationHistoryHandler_[FIT_IT_TIME_POINT].emplace_back(GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(FIT_IT_TIME_POINT));
    LogDebug << "Call " << _nbFitCalls_ << " took: "
             << GenericToolbox::parseTimeUnit(_durationHistoryHandler_[FIT_IT_TIME_POINT].back())
             << " (average: "
             << GenericToolbox::parseTimeUnit(GenericToolbox::getAverage(_durationHistoryHandler_[FIT_IT_TIME_POINT]))
             << ")" << std::endl;


    ////////////////////////////////
    // The total chi2 value is returned for Minuit
    return _chi2Buffer_;

}
void ND280Fitter::UpdateFitParameterValues(const double* par){

    ////////////////////////////////
    // Update fit parameter values for Minuit array (par)
    ////////////////////////////////

    int iMinuitParIndex = 0;
    for(int iSystSource = 0; iSystSource < _fitParametersGroupList_.size(); iSystSource++){
        for(int iParameter = 0; iParameter < _fitParametersGroupList_[iSystSource]->GetNpar(); iParameter++){
            _newParametersList_[iSystSource][iParameter] = par[iMinuitParIndex];
            iMinuitParIndex++;
        }

        if(_fitParametersGroupList_[iSystSource]->IsDecomposed()){
            _newParametersList_[iSystSource] = _fitParametersGroupList_[iSystSource]->GetOriginalParameters(
                    _newParametersList_[iSystSource]
                );
        }
    }

}
void ND280Fitter::PropagateSystematics(){

    // Time monitoring
    for(size_t jGroup = 0 ; jGroup < _fitParametersGroupList_.size() ; jGroup++){
        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads() ; iThread++ ){
            _durationReweightParameters_[jGroup][iThread].emplace_back(0);
        }
    }

    ////////////////////////////////
    // Reweight every event
    ////////////////////////////////
    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(REWEIGHT_TIME_POINT);

    if(GlobalVariables::getNbThreads() > 1){
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++){
            // Start every waiting thread
            _triggerReweightThreads_[iThread] = true;
        }

        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++){
            // triggerReweight are set to false when the thread is finished
            while(_triggerReweightThreads_[iThread]){
                // Disabled throttling -> slowing down otherwise
//                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
    }
    else{
        this->ReWeightEvents();
    }

    // Time monitoring
    _durationHistoryHandler_[REWEIGHT_TIME_POINT].emplace_back(
        GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(REWEIGHT_TIME_POINT)
        );
    if(_printFitState_){
        for(size_t jGroup = 0 ; jGroup < _fitParametersGroupList_.size() ; jGroup++){

            long long cpuTimeSpent = 0;
            long long averageTimeSpent = 0;
            for( int iThread = 0 ; iThread < TMath::Abs(GlobalVariables::getNbThreads()) ; iThread++ )
            {
                cpuTimeSpent += _durationReweightParameters_[jGroup][iThread].back();
                averageTimeSpent += GenericToolbox::getAverage(_durationReweightParameters_[jGroup][iThread]); // average over all instances
            }

            LogTrace << _fitParametersGroupList_[jGroup]->GetName() << " -> " << GenericToolbox::parseTimeUnit(cpuTimeSpent/TMath::Abs(GlobalVariables::getNbThreads()));
            LogTrace << " (average: " << GenericToolbox::parseTimeUnit(averageTimeSpent /TMath::Abs(GlobalVariables::getNbThreads())) << ")" << std::endl;
        }
        LogTrace << "Reweighting took: "
                 << GenericToolbox::parseTimeUnit(
                        _durationHistoryHandler_[REWEIGHT_TIME_POINT].back())
                 << " (average: "
                 << GenericToolbox::parseTimeUnit(GenericToolbox::getAverage(_durationHistoryHandler_[REWEIGHT_TIME_POINT]))
                 << ")"
                 << std::endl;
    }


    ////////////////////////////////
    // Re-fill the fitted MC histograms
    ////////////////////////////////
    GenericToolbox::getElapsedTimeSinceLastCallStr(FILL_TIME_POINT);

    for(int iSample = 0 ; iSample < _samplesList_.size() ; iSample++){
        if(_samplesList_[iSample]->GetPredHisto() != nullptr)
            _samplesList_[iSample]->GetPredHisto()   ->Reset("ICESM");
        if(_samplesList_[iSample]->GetMCHisto() != nullptr)
            _samplesList_[iSample]->GetMCHisto()     ->Reset("ICESM");
        if(_samplesList_[iSample]->GetMCTruthHisto() != nullptr)
            _samplesList_[iSample]->GetMCTruthHisto()->Reset("ICESM");
        if(_samplesList_[iSample]->GetSignalHisto() != nullptr)
            _samplesList_[iSample]->GetSignalHisto() ->Reset("ICESM");
    }

    if(GlobalVariables::getNbThreads() > 1){
        // Starting reweighting threads
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++){
            _triggerReFillMcHistogramsThreads_[iThread] = true;
        }

        // Wait for threads to finish their jobs
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++){
            while(_triggerReFillMcHistogramsThreads_[iThread]){ // triggerReweight are set to false when the thread is finished
//                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
    } // _nbThreads_ > 1
    else {
        // single core
        this->ReFillSampleMcHistograms();
    }

    for(int iSample = 0 ; iSample < _samplesList_.size() ; iSample++){
        if(_samplesList_[iSample]->GetPredHisto() != nullptr)
            _samplesList_[iSample]->GetPredHisto()   ->Scale(_samplesList_[iSample]->GetNorm());
        if(_samplesList_[iSample]->GetMCHisto() != nullptr)
            _samplesList_[iSample]->GetMCHisto()     ->Scale(_samplesList_[iSample]->GetNorm());
        if(_samplesList_[iSample]->GetMCTruthHisto() != nullptr)
            _samplesList_[iSample]->GetMCTruthHisto()->Scale(_samplesList_[iSample]->GetNorm());
        if(_samplesList_[iSample]->GetSignalHisto() != nullptr)
            _samplesList_[iSample]->GetSignalHisto() ->Scale(_samplesList_[iSample]->GetNorm());
    }

    _durationHistoryHandler_[FILL_TIME_POINT].emplace_back(
        GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(FILL_TIME_POINT)
    );
    if(_printFitState_){
        LogTrace << "Filling MC histograms took: "
                 << GenericToolbox::parseTimeUnit(_durationHistoryHandler_[FILL_TIME_POINT].back())
                 << " (average: "
                 << GenericToolbox::parseTimeUnit(GenericToolbox::getAverage(_durationHistoryHandler_[FILL_TIME_POINT]))
                 << ")"
                 << std::endl;
    }

}
void ND280Fitter::ComputeChi2(){

    ////////////////////////////////
    // Compute chi2 stat
    ////////////////////////////////
    for(auto & sampleContainer : _samplesList_){

        //double sampleChi2Stat = _samplesList_.at(sampleContainer)->CalcChi2();
        double sampleChi2Stat = sampleContainer->CalcLLH();
        //double sampleChi2Stat = _samplesList_.at(sampleContainer)->CalcEffLLH();

        if(_printFitState_){
            LogInfo << "Chi2 stat for sample " << sampleContainer->GetName() << " is "
                    << sampleChi2Stat << std::endl;
        }

        _chi2StatBuffer_ += sampleChi2Stat;
    }


    ////////////////////////////////
    // Compute the penalty terms
    ////////////////////////////////
    for(size_t iGroup = 0; iGroup < _fitParametersGroupList_.size(); iGroup++){

        if(not _disableChi2Pulls_){
            double systChi2Pull = _fitParametersGroupList_[iGroup]->GetChi2(_newParametersList_[iGroup]);
            _chi2PullsBuffer_ += systChi2Pull;
            if(_printFitState_){
                LogInfo << "Chi2 contribution from " << _fitParametersGroupList_[iGroup]->GetName() << " is "
                        << systChi2Pull
                        << std::endl;
            }
        }

        if(_fitParametersGroupList_[iGroup]->IsRegularised()){
            _chi2RegBuffer_ += _fitParametersGroupList_[iGroup]->CalcRegularisation(_newParametersList_[iGroup]);
        }

    }

}

// Multi-thread compatible methods
void ND280Fitter::ReWeightEvents(int iThread_){

    AnaEvent* eventPtr;
    bool isMultiThreaded = (iThread_ != -1);
    std::string progressTitle; // single core only

    // Looping over all events
    for(int iSample = 0; iSample < _samplesList_.size(); iSample++){
        for(int iEvent = 0; iEvent < _samplesList_.at(iSample)->GetN(); iEvent++){

            if(isMultiThreaded){
                if( iEvent % GlobalVariables::getNbThreads() != iThread_){
                    continue; // skip this event
                }
            }

            eventPtr = _samplesList_.at(iSample)->GetEvent(iEvent);
            eventPtr->ResetEvWght();

            for(size_t jGroup = 0 ; jGroup < _fitParametersGroupList_.size() ; jGroup++){
                GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+iThread_);
                _fitParametersGroupList_.at(jGroup)->ReWeight(
                    eventPtr, _samplesList_.at(iSample)->GetDetector(),
                    iSample, iEvent,_newParametersList_.at(jGroup)
                );
                if(isMultiThreaded) _durationReweightParameters_[jGroup][iThread_].back() += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+iThread_);
                else _durationReweightParameters_[jGroup][0].back() += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+iThread_);
            } // jParam

        } // iEvent

    } // iSample

    // For the cache at the next reweight call
    for(size_t jGroup = 0 ; jGroup < _fitParametersGroupList_.size() ; jGroup++){
        _fitParametersGroupList_.at(jGroup)->SetLastAppliedParamList(_newParametersList_.at(jGroup));
    }

}
void ND280Fitter::ReFillSampleMcHistograms(int iThread_){

    bool isMultiThreaded = (iThread_ != -1);

//    GlobalVariables::getThreadMutex().lock();
    TH1D* histPredPtr = nullptr;
    TH1D* histMcPtr = nullptr;
    TH1D* histMcTruePtr = nullptr;
    TH1D* histSigPtr = nullptr;
    AnaEvent* anaEventPtr = nullptr;
    double eventWeightBuffer;
//    GlobalVariables::getThreadMutex().unlock();

    for(int iSample = 0 ; iSample < _samplesList_.size() ; iSample++){

        if(isMultiThreaded){
            histPredPtr   = _histThreadHandlers_[iThread_][iSample][0];
            histMcPtr     = _histThreadHandlers_[iThread_][iSample][1];
            histMcTruePtr = _histThreadHandlers_[iThread_][iSample][2];
            histSigPtr    = _histThreadHandlers_[iThread_][iSample][3];
        }
        else{
            histPredPtr   = _samplesList_[iSample]->GetPredHisto();
            histMcPtr     = _samplesList_[iSample]->GetMCHisto();
            histMcTruePtr = _samplesList_[iSample]->GetMCTruthHisto();
            histSigPtr    = _samplesList_[iSample]->GetSignalHisto();
        }

        if(histPredPtr != nullptr)   histPredPtr->Reset("ICESM");
        if(histMcPtr != nullptr)     histMcPtr->Reset("ICESM");
        if(histMcTruePtr != nullptr) histMcTruePtr->Reset("ICESM");
        if(histSigPtr != nullptr)    histSigPtr->Reset("ICESM");

        for(int iEvent = 0 ; iEvent < _samplesList_[iSample]->GetN() ; iEvent++){

            if(isMultiThreaded){
                if( iEvent%GlobalVariables::getNbThreads() != iThread_){
                   continue;
                }
            }

            anaEventPtr = _samplesList_[iSample]->GetEvent(iEvent);

            // Events are not supposed to move for one bin to another with the current implementation
            // So the bin index shall be computed once
            if(anaEventPtr->GetTrueBinIndex() == -1){
                anaEventPtr->GetTrueBinIndex() = _samplesList_[iSample]->GetBinIndex(
                    anaEventPtr->GetTrueD1(),
                    anaEventPtr->GetTrueD2()
                );
            }

            if(anaEventPtr->GetRecoBinIndex() == -1){
                anaEventPtr->GetRecoBinIndex() = _samplesList_[iSample]->GetBinIndex(
                    anaEventPtr->GetRecoD1(),
                    anaEventPtr->GetRecoD2()
                );
            }

            eventWeightBuffer = anaEventPtr->GetEvWght();

            if(histPredPtr != nullptr) histPredPtr->Fill(
                anaEventPtr->GetRecoBinIndex() + 0.5, eventWeightBuffer
            );
            if(histMcPtr != nullptr) histMcPtr->Fill(
                anaEventPtr->GetRecoBinIndex() + 0.5, eventWeightBuffer
            );
            if(histMcTruePtr != nullptr) histMcTruePtr->Fill(
                anaEventPtr->GetTrueBinIndex() + 0.5, eventWeightBuffer
            );

            if(anaEventPtr->isSignalEvent()){
                if(histSigPtr != nullptr) histSigPtr->Fill(
                    anaEventPtr->GetTrueBinIndex() + 0.5, eventWeightBuffer
                );
            }

        }

        if(isMultiThreaded){
            GlobalVariables::getThreadMutex().lock();
            if(_samplesList_[iSample]->GetPredHisto() != nullptr)
                _samplesList_[iSample]->GetPredHisto()->Add(histPredPtr);
            if(_samplesList_[iSample]->GetMCHisto() != nullptr)
                _samplesList_[iSample]->GetMCHisto()->Add(histMcPtr);
            if(_samplesList_[iSample]->GetMCTruthHisto() != nullptr)
                _samplesList_[iSample]->GetMCTruthHisto()->Add(histMcTruePtr);
            if(_samplesList_[iSample]->GetSignalHisto() != nullptr)
                _samplesList_[iSample]->GetSignalHisto()->Add(histSigPtr);
            GlobalVariables::getThreadMutex().unlock();
        }

    }

}


void ND280Fitter::ParameterScans(const std::vector<int>& param_list, unsigned int nsteps)
{
    LogInfo << "Performing parameter scans..." << std::endl;

    //Internally Scan performs steps-1, so add one to actually get the number of steps
    //we ask for.
    unsigned int adj_steps = nsteps+1;
    double* x = new double[adj_steps] {};
    double* y = new double[adj_steps] {};

    for(const auto& p : param_list)
    {
        LogInfo << "Scanning parameter " << p
                  << " (" << _minimizer_->VariableName(p) << ")." << std::endl;

        bool success = _minimizer_->Scan(p, adj_steps, x, y);

        TGraph scan_graph(nsteps, x, y);
        _outputTDirectory_->cd();

        std::stringstream ss;
        ss << "par_scan_" << std::to_string(p);
        scan_graph.Write(ss.str().c_str());
    }

    delete[] x;
    delete[] y;
}

void ND280Fitter::SaveEventHist(int fititer, bool is_final)
{
    for(int s = 0; s < _samplesList_.size(); s++)
    {
        std::stringstream ss;
        ss << "evhist_sam" << s;
        if(is_final)
            ss << "_finaliter";
        else
            ss << "_iter" << _nbFitCalls_;

        _samplesList_[s]->Write(_outputTDirectory_, ss.str(), fititer);
    }
}
void ND280Fitter::SaveEventTree(std::vector<std::vector<double>>& res_params)
{
    outtree = new TTree("selectedEvents", "selectedEvents");
    InitOutputTree();

    for(size_t s = 0; s < _samplesList_.size(); s++)
    {
        for(int i = 0; i < _samplesList_[s]->GetN(); i++)
        {
            AnaEvent* ev = _samplesList_[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            for(size_t j = 0; j < _fitParametersGroupList_.size(); j++)
            {
                const std::string det = _samplesList_[s]->GetDetector();
                _fitParametersGroupList_[j]->ReWeight(ev, det, s, i, res_params[j]);
            }

            sample   = ev->GetSampleType();
            sigtype  = ev->GetSignalType();
            topology = ev->GetTopology();
            reaction = ev->GetReaction();
            target   = ev->GetTarget();
            nutype   = ev->GetFlavor();
            D1true   = ev->GetTrueD1();
            D2true   = ev->GetTrueD2();
            D1Reco   = ev->GetRecoD1();
            D2Reco   = ev->GetRecoD2();
            weightMC = ev->GetEvWghtMC() * _MC_normalization_factor_;
            weight   = ev->GetEvWght()   * _MC_normalization_factor_;
            outtree->Fill();
        }
    }
    _outputTDirectory_->cd();
    outtree->Write();
}
void ND280Fitter::SaveParams(const std::vector<std::vector<double>>& new_pars)
{
    std::vector<double> temp_vec;
    for(size_t i = 0; i < _fitParametersGroupList_.size(); i++)
    {
        const unsigned int npar = _fitParametersGroupList_[i]->GetNpar();
        const std::string name  = _fitParametersGroupList_[i]->GetName();
        std::stringstream ss;

        ss << "hist_" << name << "_iter" << _nbFitCalls_;
        TH1D h_par(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        std::vector<std::string> vec_names;
        _fitParametersGroupList_[i]->GetParNames(vec_names);
        for(int j = 0; j < npar; j++)
        {
            h_par.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par.SetBinContent(j + 1, new_pars[i][j]);
            temp_vec.emplace_back(new_pars[i][j]);
        }
        _outputTDirectory_->cd();
        h_par.Write();
    }

    TVectorD root_vec(temp_vec.size(), &temp_vec[0]);
    root_vec.Write(Form("vec_par_all_iter%d", _nbFitCalls_));
}
void ND280Fitter::SaveResults(const std::vector<std::vector<double>>& par_results,
                              const std::vector<std::vector<double>>& par_errors)
{
    for(std::size_t i = 0; i < _fitParametersGroupList_.size(); i++)
    {
        const unsigned int npar = _fitParametersGroupList_[i]->GetNpar();
        const std::string name  = _fitParametersGroupList_[i]->GetName();
        std::vector<double> par_original;
        _fitParametersGroupList_[i]->GetParOriginal(par_original);

        TMatrixDSym* cov_mat = _fitParametersGroupList_[i]->GetOriginalCovMat();

        std::stringstream ss;

        ss << "hist_" << name << "_result";
        TH1D h_par_final(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_prior";
        TH1D h_par_prior(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_error_final";
        TH1D h_err_final(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_error_prior";
        TH1D h_err_prior(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        std::vector<std::string> vec_names;
        _fitParametersGroupList_[i]->GetParNames(vec_names);
        for(int j = 0; j < npar; j++)
        {
            h_par_final.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par_final.SetBinContent(j + 1, par_results[i][j]);
            h_par_prior.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par_prior.SetBinContent(j + 1, par_original[j]);
            h_err_final.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_err_final.SetBinContent(j + 1, par_errors[i][j]);

            double err_prior = 0.0;
            if(cov_mat != nullptr)
                err_prior = TMath::Sqrt((*cov_mat)(j,j));

            h_err_prior.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_err_prior.SetBinContent(j + 1, err_prior);
        }

        _outputTDirectory_->cd();
        h_par_final.Write();
        h_par_prior.Write();
        h_err_final.Write();
        h_err_prior.Write();
    }
}

TMatrixD* ND280Fitter::GeneratePriorCovarianceMatrix(){

  std::vector<TMatrixD*> matrix_category_list;
  int nb_dof = 0;
  for(int i_parameter = 0 ; i_parameter < _fitParametersGroupList_.size() ; i_parameter++){
    nb_dof += _fitParametersGroupList_[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  auto* covMatrix = new TMatrixD(nb_dof,nb_dof);

  int index_shift = 0;
  for(int i_parameter = 0 ; i_parameter < _fitParametersGroupList_.size() ; i_parameter++){
    for(int i_entry = 0 ; i_entry < _fitParametersGroupList_[i_parameter]->GetOriginalCovMat()->GetNrows() ; i_entry++){
      for(int j_entry = 0 ; j_entry < _fitParametersGroupList_[i_parameter]->GetOriginalCovMat()->GetNrows() ; j_entry++){
        (*covMatrix)[i_entry+index_shift][j_entry+index_shift] = (*_fitParametersGroupList_[i_parameter]->GetOriginalCovMat())[i_entry][j_entry] ;
      }
    }
    index_shift += _fitParametersGroupList_[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  return covMatrix;

}
TMatrixD* ND280Fitter::GeneratePosteriorCovarianceMatrix(){

  auto* covMatrix = new TMatrixD(_minimizer_->NDim(), _minimizer_->NDim() );
  _minimizer_->GetCovMatrix( covMatrix->GetMatrixArray() );
  return covMatrix;

}

