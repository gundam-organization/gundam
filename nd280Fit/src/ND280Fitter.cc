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

    _disableMultiThread_ = false;

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

    // Internals
    _isInitialized_     = false;
    _fitHasBeenDone_    = false;
    _fitHasConverged_   = false;


    if(GlobalVariables::getNbThreads() > 1 and _stopThreads_ == false){
        LogInfo << "Stopping parallel threads..." << std::endl;
        _stopThreads_ = true;
        // the following provoques Segfault
//        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads() ; iThread++ ){
//            _asyncFitThreads_[iThread].get(); // Waiting for the end of each thread
//        }
    }

    _advancedTimeMonitoring_ = false;

    _PRNG_seed_ = -1;

    delete _minimizer_;         _minimizer_ = nullptr;
    delete _functor_;           _functor_ = nullptr;

    _outputTDirectory_ = nullptr; // not handle by this class

    _saveFitParameters_                         = false;
    _saveEventTree_                             = true;
    _disableChi2Pulls_                          = false;
    _apply_statistical_fluctuations_on_samples_ = false;

    _saveFitParamsFrequency_     = 10000;
    _nb_fit_parameters_          = 0;
    _nbFitCalls_                 = 0;
    _nbTotalEvents_              = 0;

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
void ND280Fitter::SetMinimizationSettings(const MinSettings& minimization_settings_){
    _minimization_settings_ = minimization_settings_;
    if(_minimizer_ != nullptr){
        // If the fitter has already been initialized, set these parameters
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
void ND280Fitter::SetSaveEventTree(bool save_event_tree_){ _saveEventTree_ = save_event_tree_;
}
void ND280Fitter::SetSaveFitParams(bool save_fit_params_){
    _saveFitParameters_ = save_fit_params_;
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
    InitializeFitter();
    InitializeDataSamples();

    LogInfo << "Saving snapshot of samples state..." << std::endl;
    for(const auto& sample : _samplesList_){
        sample->SaveMcEventsSnapshot();
        sample->SaveHistogramsSnapshot();
    }

    _isInitialized_ = true;
    LogWarning << "ND280 Fitter has been initialized." << std::endl;

}


void ND280Fitter::WritePrefitData(){

    LogInfo << "Writing prefit data..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    if(_fitHasBeenDone_){
        LogFatal << "Can't write prefit data when fit has been performed." << std::endl;
        throw std::logic_error(GET_VAR_NAME_VALUE(_fitHasBeenDone_).c_str());
    }

    auto* preFitDir = GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit");

    /////////////////////////////
    // Samples
    /////////////////////////////
    this->WriteSamplePlots(preFitDir);


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

        GenericToolbox::mkdirTFile(_outputTDirectory_, Form("prefit/parameters/%s", anaFitParameters->GetName().c_str()))->cd();

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

    GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit/parameters")->cd();

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

    LogInfo << "Copying current parameters..." << std::endl;
    auto* par = new double[_minimizer_->NDim()];
    for(unsigned int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++){
        par[iPar] = _minimizer_->X()[iPar];
    }

    // Make sure all weights a applied
    LogInfo << "Re-computing weights..." << std::endl;
    UpdateFitParameterValues(par);
    PropagateSystematics();
    ComputeChi2();

    double originalChi2 = _chi2Buffer_;

    bool stopNow = false;
    for(const auto& sample : _samplesList_){
        double llh = sample->CalcLLH();
        if(llh != 0){
            sample->CompareMcEventsWeightsWithSnapshot();
            sample->CompareHistogramsWithSnapshot();
            LogDebug << GET_VAR_NAME_VALUE(llh) << std::endl;
            stopNow = true;
        }
    }
    if(stopNow){
        _disableMultiThread_ = true;
        LogError << "RETRYING SINGLE THREAD?" << std::endl;
        this->PropagateSystematics();
        for(const auto& sample : _samplesList_){
            double llh = sample->CalcLLH();
            LogDebug << GET_VAR_NAME_VALUE(llh) << std::endl;
        }
        throw std::runtime_error("STOP");
    }

    LogInfo << "Computing nominal MC histograms..." << std::endl;
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
        for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
            auto* anaEvent = &anaSample->GetMcEvents().at(iEvent);
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
        for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
            auto* anaEvent = &anaSample->GetMcEvents().at(iEvent);
            nominalHistMap[anaSample->GetName() + "/D1"]->Fill( anaEvent->GetRecoD1(), anaEvent->GetEvWght() );
        }

    }

    LogInfo << "Performing individual one sigma variations..." << std::endl;
    TMatrixD* covarianceMatrix = this->GeneratePriorCovarianceMatrix();
    std::vector<std::map<std::string, TH1D*>> oneSigmaHistMap(_minimizer_->NDim());
    for( int iSyst = 0 ; iSyst < _minimizer_->NDim() ; iSyst++ ){

        // If this parameter is fixed, then it won't be writen
        if(_fitParameterIsFixedList_[iSyst]){
            LogTrace << "Skipping fixed parameter: " << iSyst << std::endl;
            continue;
        }

        LogInfo << "Running +1 sigma on: " << _parameter_names_[iSyst] << std::endl;
        // which systematic is it?
        int parCount = 0;
        int parGroupIndex = -1;
        int parIndex = -1;
        for( size_t iParGroup = 0 ; iParGroup < _fitParametersGroupList_.size(); iParGroup++ ){
            for( size_t iPar = 0 ; iPar < _fitParametersGroupList_[iParGroup]->GetNpar() ; iPar++ ){
                if(parCount == iSyst){
                    parIndex = iPar;
                    parGroupIndex = iParGroup;
                    break;
                }
                parCount++;
            }
            if(parIndex != -1) break;
        }
        _fitParametersGroupList_[parGroupIndex]->PrintParameterInfo(parIndex);

        // Create associated subdirectory
        auto* currentDir = GenericToolbox::mkdirTFile( oneSigmaDir, _parameter_names_[iSyst] );
        currentDir->cd();

        // Put the parameter at 1 sigma
        double originalPar = par[iSyst];
        par[iSyst] += TMath::Sqrt((*covarianceMatrix)[iSyst][iSyst]);
        UpdateFitParameterValues(par);
        PropagateSystematics();

//        ComputeChi2();
//        double chi2 = _chi2Buffer_;

        // Propagate systematics

        std::vector<Color_t> colorWheel = {
            kGreen-3,
            kTeal+3,
            kAzure+7,
            kCyan-2,
            kBlue-7,
            kBlue+2
        };
        int colorTick = 0;

        bool parameterHasNoEffect = true;       // No need to fit this parameter since the samples give no response
        bool parameterIsDegenerated = true;     // Don't fit with multiple parameters that affect the sample in the same way

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
            for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
                auto* anaEvent = &anaSample->GetMcEvents().at(iEvent);
                oneSigmaHistMap[iSyst][tempHistName]->Fill( anaEvent->GetRecoD1(), anaEvent->GetEvWght() );
            }

            // Is affected?
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
                LogWarning << anaSample->GetName() << " is unaffected by \"" << _parameter_names_[iSyst] << "\"" << std::endl;
            }
            else{
                parameterHasNoEffect = false;
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

        if(parameterHasNoEffect){
            LogError << "Parameter: \"" << _parameter_names_[iSyst] << "\" has no effect on the samples, fixing it for the fit." << std::endl;
            _fitParameterGhostList_.emplace_back(iSyst);
            _fitParameterIsFixedList_[iSyst] = true;
            _minimizer_->FixVariable(iSyst);
        }
        else{
            // Is degenerated?
            for( size_t jSyst = 0 ; jSyst < iSyst ; jSyst++ ){
                bool hasSameEffect = true;
                for( const auto& anaSample : _samplesList_){
                    std::string tempHistName = anaSample->GetName() + "/D1";
                    for(int iBin = 0 ; iBin <= oneSigmaHistMap[iSyst][tempHistName]->GetNbinsX() + 1 ; iBin++){
                        if( TMath::Abs(oneSigmaHistMap[jSyst][tempHistName]->GetBinContent(iBin) - oneSigmaHistMap[iSyst][tempHistName]->GetBinContent(iBin)) > 1E-6 ){
                            hasSameEffect = false;
                            break;
                        }
                    }
                    if(not hasSameEffect) break;
                }
                if(hasSameEffect){
                    LogError << _parameter_names_[iSyst] << " has the same effect as " << _parameter_names_[jSyst] << " on samples." << std::endl;
                    LogError << "\"" << _parameter_names_[iSyst] << "\" will be fixed in the fit." << std::endl;
                    _fitParameterDegeneracyList_[iSyst] = jSyst;
                    _fitParameterIsFixedList_[iSyst] = true;
                    _minimizer_->FixVariable(iSyst);
                    break;
                }

            }
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

        LogDebug << "Coming back to 0 sigmas..." << std::endl;
        par[iSyst] = originalPar;
        UpdateFitParameterValues(par);
        PropagateSystematics();
        _printFitState_ = false;
        ComputeChi2();
        _printFitState_ = true;
        if(originalChi2 != _chi2Buffer_){
            for(const auto& sample : _samplesList_){
                double llh = sample->CalcLLH();
                LogDebug << GET_VAR_NAME_VALUE(llh) << ": " << (llh == 0) << std::endl;
                if(llh != 0){
                    sample->CompareMcEventsWeightsWithSnapshot();
                    sample->CompareHistogramsWithSnapshot();
                    throw std::runtime_error("lol");
                }
            }
        }

        LogDebug << std::endl; // aesthetic

    }

    delete[] par;
    _outputTDirectory_->cd();

    LogInfo << "Make one sigma check just ended." << std::endl;

}
bool ND280Fitter::Fit(){

    LogAlert << "Starting to fit." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    if(_selected_data_type_ == kAsimov){

        LogWarning << "Asimov fit detected, applying slight shift on the prior parameters..." << std::endl;
        TMatrixD* covarianceMatrix = this->GeneratePriorCovarianceMatrix();
        for( int iPar = 0; iPar < _nb_fit_parameters_; iPar++ ) {

            if(not _fitParameterIsFixedList_[iPar]){
                _minimizer_->SetVariable(
                    iPar, _parameter_names_[iPar],
                    _parameterPriorValues_[iPar] + _PRNG_->Gaus(0, 0.01*TMath::Sqrt((*covarianceMatrix)[iPar][iPar])),
                    _parameterSteps_[iPar]
                );
            }

        }

        // Make sure all weights a applied
        UpdateFitParameterValues(_minimizer_->X());
        PropagateSystematics();
        ComputeChi2();

        WriteSamplePlots(GenericToolbox::mkdirTFile(_outputTDirectory_, "fit")); // just to control the first iteration...
        
    }

    LogInfo << "List of all fit parameters: " << std::endl;
    int paramaterOffset = 0;
    for( size_t iParGroup = 0 ; iParGroup < _fitParametersGroupList_.size() ; iParGroup++ ){

        LogInfo << "- " << _fitParametersGroupList_[iParGroup]->GetName() << ": " << std::endl;
        for( size_t iPar = 0 ; iPar < _fitParametersGroupList_[iParGroup]->GetNpar() ; iPar++ ){

            if(_fitParameterIsFixedList_[paramaterOffset + iPar]){

//                _fitParametersGroupList_[iParGroup]->PrintParameterInfo(iPar);
                LogAlert << _fitParametersGroupList_[iParGroup]->GetParNames()[iPar];
                if(GenericToolbox::doesElementIsInVector(paramaterOffset + iPar, _fitParameterGhostList_)){
                    LogAlert << " -> FIXED: No effect on the samples" << std::endl;
                }
                else if(GenericToolbox::doesKeyIsInMap(paramaterOffset + iPar, _fitParameterDegeneracyList_)){
                    LogAlert << " -> FIXED: had same effect as parameter: "
                             << _fitParametersGroupList_[iParGroup]->GetParNames()[_fitParameterDegeneracyList_[paramaterOffset + iPar]-paramaterOffset]
                             << std::endl;
                }
                else{
                    LogAlert << " -> FIXED" << std::endl;
                }

            }
            else{
                LogInfo << _fitParametersGroupList_[iParGroup]->GetParNames()[iPar] << std::endl;
            }

        }
        paramaterOffset += _fitParametersGroupList_[iParGroup]->GetNpar();
    }

    auto* systPostFitDir = GenericToolbox::mkdirTFile(_outputTDirectory_, "prefit/parameters");

    LogWarning << "Scanning parameters arround the prefit point..." << std::endl;
    paramaterOffset = 0;
    for( size_t iParGroup = 0 ; iParGroup < _fitParametersGroupList_.size() ; iParGroup++ ){

        auto* samplePostFitDir = GenericToolbox::mkdirTFile(systPostFitDir, _fitParametersGroupList_[iParGroup]->GetName());
        samplePostFitDir->cd();

        // SCAN
        unsigned int nbSteps_ = 20;
        auto* x = new double[nbSteps_] {};
        auto* y = new double[nbSteps_] {};
        GenericToolbox::mkdirTFile(samplePostFitDir, "scans")->cd();
        for( size_t iPar = 0 ; iPar < _fitParametersGroupList_[iParGroup]->GetNpar() ; iPar++ ){

            if(_fitParameterIsFixedList_[paramaterOffset + iPar]) continue;

            LogWarning << "Scanning parameter " << iPar;
            LogWarning << " (" << _minimizer_->VariableName(iPar) << ")..." << std::endl;

            bool isSuccess = _minimizer_->Scan(paramaterOffset + iPar, nbSteps_, x, y);

            TGraph scanGraph(nbSteps_, x, y);
            scanGraph.Write( (_fitParametersGroupList_[iParGroup]->GetParNames()[iPar] + "_TGraph").c_str() );

        }
        delete[] x;
        delete[] y;
        paramaterOffset += _fitParametersGroupList_[iParGroup]->GetNpar();

    }
    _outputTDirectory_->cd();


    LogAlert << "Calling Minimize, running: " << _minimization_settings_.algorithm << std::endl;
    // Run the actual fitter:
    _fitHasStarted_ = true;
    _fitHasConverged_ = _minimizer_->Minimize();

    if(not _fitHasConverged_){
        LogError << "Fit did not converge while running " << _minimization_settings_.algorithm << std::endl;
        LogError << "Failed with status code: " << _minimizer_->Status() << std::endl;
    }
    else{
        LogInfo << "Fit converged." << std::endl
                << "Status code: " << _minimizer_->Status() << std::endl;

        LogInfo << "Calling HESSE." << std::endl;
        _fitHasConverged_ = _minimizer_->Hesse();

        if(not _fitHasConverged_){
            LogError  << "Hesse did not converge." << std::endl;
            LogError  << "Failed with status code: " << _minimizer_->Status() << std::endl;
        }
        else{
            LogInfo << "Hesse converged." << std::endl
                    << "Status code: " << _minimizer_->Status() << std::endl;
        }

    }

    _fitHasBeenDone_ = true;

    LogWarning << "Fit has ended." << std::endl;
    return _fitHasConverged_;

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


    auto* postFitDir = GenericToolbox::mkdirTFile(_outputTDirectory_, "postfit");
    this->WriteSamplePlots(postFitDir);
    postFitDir->cd();

    auto* systPostFitDir = GenericToolbox::mkdirTFile(postFitDir, "parameters");

    LogInfo << "Scanning parameters around the minimum..." << std::endl;
    int paramaterOffset = 0;
    for( size_t iParGroup = 0 ; iParGroup < _fitParametersGroupList_.size() ; iParGroup++ ){

        auto* samplePostFitDir = GenericToolbox::mkdirTFile(systPostFitDir, _fitParametersGroupList_[iParGroup]->GetName());
        samplePostFitDir->cd();

        // SCAN
        unsigned int nbSteps_ = 20;
        auto* x = new double[nbSteps_] {};
        auto* y = new double[nbSteps_] {};
        GenericToolbox::mkdirTFile(samplePostFitDir, "scans")->cd();
        for( size_t iPar = 0 ; iPar < _fitParametersGroupList_[iParGroup]->GetNpar() ; iPar++ ){

            if(_fitParameterIsFixedList_[paramaterOffset + iPar]) continue;

            LogWarning << "Scanning parameter " << iPar;
            LogWarning << " (" << _minimizer_->VariableName(iPar) << ")..." << std::endl;

            bool isSuccess = _minimizer_->Scan(paramaterOffset + iPar, nbSteps_, x, y);

            TGraph scanGraph(nbSteps_, x, y);
            scanGraph.Write( (_fitParametersGroupList_[iParGroup]->GetParNames()[iPar] + "_TGraph").c_str() );

        }
        delete[] x;
        delete[] y;

        paramaterOffset += _fitParametersGroupList_[iParGroup]->GetNpar();

    }

    LogInfo << "Writing postfit parameters..." << std::endl;
    const int nfree = _minimizer_->NFree();
    if(_minimizer_->X() != nullptr){

        double covarianceMatrixArray[_minimizer_->NDim() * _minimizer_->NDim()];
        _minimizer_->GetCovMatrix(covarianceMatrixArray);
        TMatrixDSym covarianceMatrix(_minimizer_->NDim(), covarianceMatrixArray);

        std::vector<double> parameterValueList(_minimizer_->X(),      _minimizer_->X()      + _minimizer_->NDim());
        std::vector<double> parameterErrorList(_minimizer_->Errors(), _minimizer_->Errors() + _minimizer_->NDim());

        paramaterOffset = 0;
        for( size_t iParGroup = 0 ; iParGroup < _fitParametersGroupList_.size() ; iParGroup++ ){

            auto* samplePostFitDir = GenericToolbox::mkdirTFile(systPostFitDir, _fitParametersGroupList_[iParGroup]->GetName().c_str());
            samplePostFitDir->cd();

            auto* covMatrix = (TMatrixD*) _fitParametersGroupList_[iParGroup]->GetCovMat()->Clone(); // just for the tempate template
            for(int iComp = 0 ; iComp < _fitParametersGroupList_[iParGroup]->GetNpar() ; iComp++){
                for(int jComp = 0 ; jComp < _fitParametersGroupList_[iParGroup]->GetNpar() ; jComp++){
                    (*covMatrix)[iComp][jComp] = covarianceMatrix[paramaterOffset+iComp][paramaterOffset+jComp];
                } // jComp
            } // iComp
            auto* corMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*) covMatrix);
            auto* covMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) covMatrix, Form("Covariance_%s", _fitParametersGroupList_[iParGroup]->GetName().c_str()));
            auto* corMatrixTH2D = GenericToolbox::convertTMatrixDtoTH2D(corMatrix, Form("Correlation_%s", _fitParametersGroupList_[iParGroup]->GetName().c_str()));

            TH1D* histPostfitParametersCategory = new TH1D("histPostfitParameters","histPostfitParameters",
                                                          _nb_fit_parameters_,0, _nb_fit_parameters_);
            TH1D* histPostfitParametersError = new TH1D("histPostfitParametersError","histPostfitParametersError",
                                                       _nb_fit_parameters_,0, _nb_fit_parameters_);

            for(int iPar = 0 ; iPar < _fitParametersGroupList_[iParGroup]->GetNpar() ; ++iPar){

                // Parameter Prior Value
                histPostfitParametersCategory->SetBinContent(iPar+1, parameterValueList[paramaterOffset + iPar]);
                histPostfitParametersCategory->SetBinError(iPar+1, parameterErrorList[paramaterOffset + iPar] );

                histPostfitParametersError->SetBinContent(iPar+1, TMath::Sqrt((*covMatrix)[iPar][iPar]) );

                // Fill Vectors
//                vecPostfitParameters[parGlobalIndex -1] = anaFitParameters->GetParOriginal(iPar);
//                vecPostfitDecompParameters[parGlobalIndex -1] = anaFitParameters->GetParPrior(iPar);

                // Labels
                std::vector<std::string> vecBuffer;
                _fitParametersGroupList_[iParGroup]->GetParNames(vecBuffer);
                histPostfitParametersCategory->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                histPostfitParametersError   ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                covMatrixTH2D                ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                covMatrixTH2D                ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                corMatrixTH2D                ->GetXaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());
                corMatrixTH2D                ->GetYaxis()->SetBinLabel(iPar+1, vecBuffer[iPar].c_str());

            }

            histPostfitParametersCategory->Write("postFitParameters_TH1D");
            histPostfitParametersError->Write("postFitErrors_TH1D");
            covMatrix->Write("covarianceMatrix_TMatrixD");
            corMatrix->Write("correlationMatrix_TMatrixD");
            covMatrixTH2D->Write("covarianceMatrix_TH2D");
            corMatrixTH2D->Write("correlationMatrix_TH2D");

            samplePostFitDir->cd();


            paramaterOffset += _fitParametersGroupList_[iParGroup]->GetNpar();
        } // fit Parameters

        systPostFitDir->cd();
        covarianceMatrix.Write("covarianceMatrix_TMatrixDSym");

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
//    if(_saveEventTree_) SaveEventTree(res_pars);

    _outputTDirectory_->cd();

}
void ND280Fitter::ScanParameters(unsigned int nbSteps_){

    auto* x = new double[nbSteps_] {};
    auto* y = new double[nbSteps_] {};

    for( size_t iPar = 0 ; iPar < _parameter_names_.size() ; iPar++ ){

        LogWarning << "Scanning parameter " << iPar;
        LogWarning << " (" << _minimizer_->VariableName(iPar) << ")..." << std::endl;

        bool isSuccess = _minimizer_->Scan(iPar, nbSteps_, x, y);

        TGraph scanGraph(nbSteps_, x, y);
        GenericToolbox::mkdirTFile(_outputTDirectory_, "postfit")->cd();

        std::stringstream ss;
        ss << "par_scan_" << std::to_string(iPar);
        scanGraph.Write(ss.str().c_str());

    }

    _outputTDirectory_->cd();

    delete[] x;
    delete[] y;

}

void ND280Fitter::WriteSamplePlots(TDirectory* outputDir_) {

    LogWarning << "Generating sample plots." << std::endl;

    auto* samplesDir = GenericToolbox::mkdirTFile(outputDir_, "samples");
    LogInfo << "Samples plots will be writen in: " << samplesDir->GetPath() << std::endl;

//    double maxD1Scale = 2000;
    double maxD1Scale = -1;

    // TODO: do the same thing for histograms

    // Select which canvas to plot
    std::vector<std::string> canvasSubFolderList;
    canvasSubFolderList.emplace_back("Raw");
    canvasSubFolderList.emplace_back("Raw/reactions");
    canvasSubFolderList.emplace_back("D1");             // varToPlot
    canvasSubFolderList.emplace_back("D1/reactions");   // varToPlot/splitHist
    canvasSubFolderList.emplace_back("D2");             // varToPlot
    canvasSubFolderList.emplace_back("D2/reactions");   // varToPlot/splitHist

    std::vector<Color_t> colorWheel = {
        kGreen-3, kTeal+3, kAzure+7,
        kCyan-2, kBlue-7, kBlue+2,
        kOrange+1, kOrange+9, kRed+2, kPink+9
    };

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


    LogInfo << "Generating and writing sample histograms..." << std::endl;
    std::map<std::string, TH1D*> TH1D_handler;
    std::map<std::string, int> splitVarColor;
    int sampleColor = 0;
    samplesDir->cd();
    for(const auto& anaSample : _samplesList_){

        std::map<std::string, TH1D*> tempHistMap;

        LogDebug << "Processing histograms for: " << anaSample->GetName() << std::endl;

        std::string histNameBuffer;

        // Sample's raw histograms (what's actually fitted)
        histNameBuffer              = anaSample->GetName() + "/Raw/MC";
        tempHistMap[histNameBuffer] = (TH1D*) anaSample->GetPredHisto().Clone();
        tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
        tempHistMap[histNameBuffer]->SetTitle("MC");
        histNameBuffer              = anaSample->GetName() + "/Raw/Data";
        tempHistMap[histNameBuffer] = (TH1D*) anaSample->GetDataHisto().Clone();
        tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
        tempHistMap[histNameBuffer]->SetTitle("Data");

        // Build the histograms binning (D1)
        // TODO: check if a simple binning can be applied
        bool isSimpleBinning = true;
        std::vector<double> D1binning;
        auto bins = anaSample->GetBinEdges();
        for(const auto& bin : bins){
            if(D1binning.empty()) D1binning.emplace_back(bin.D1low);
            if(not GenericToolbox::doesElementIsInVector(bin.D1high, D1binning)){
                D1binning.emplace_back(bin.D1high);
            }
        }
        std::sort(D1binning.begin(), D1binning.end());

        histNameBuffer              = anaSample->GetName() + "/D1/MC";
        tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "MC",
                                             D1binning.size() - 1, &D1binning[0]);
        histNameBuffer              = anaSample->GetName() + "/D1/Data";
        tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "Data",
                                             D1binning.size() - 1, &D1binning[0]);

        // Build the histograms binning (D2)
        // TODO: check if a simple binning can be applied
        isSimpleBinning = true;
        std::vector<double> D2binning;
        for(const auto& bin : bins){
            if(D2binning.empty()) D2binning.emplace_back(bin.D2low);
            if(not GenericToolbox::doesElementIsInVector(bin.D2high, D2binning)){
                D2binning.emplace_back(bin.D2high);
            }
        }
        std::sort(D2binning.begin(), D2binning.end());

        histNameBuffer              = anaSample->GetName() + "/D2/MC";
        tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "MC",
                                               D2binning.size() - 1, &D2binning[0]);
        histNameBuffer              = anaSample->GetName() + "/D2/Data";
        tempHistMap[histNameBuffer] = new TH1D(histNameBuffer.c_str(), "Data",
                                               D2binning.size() - 1, &D2binning[0]);

        // Get the list of valid sub-divisions...
        {   // Reaction
            splitVarColor["reactions"] = 0;
            std::vector<int> reactionCodesList;
            for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
                auto* anaEvent = anaSample->GetEvent(iEvent);
                if(not GenericToolbox::doesElementIsInVector(anaEvent->GetReaction(),
                                                             reactionCodesList)){
                    reactionCodesList.emplace_back(anaEvent->GetReaction());
                }
            }
            for(const auto& thisReactionCode : reactionCodesList){
                histNameBuffer = anaSample->GetName() + "/D1/reactions/" + std::to_string(thisReactionCode);
                tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/D1/MC"]->Clone();
                tempHistMap[histNameBuffer]->Reset("ICESM");
                tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
                tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());

                histNameBuffer = anaSample->GetName() + "/D2/reactions/" + std::to_string(thisReactionCode);
                tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/D2/MC"]->Clone();
                tempHistMap[histNameBuffer]->Reset("ICESM");
                tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
                tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());

                histNameBuffer = anaSample->GetName() + "/Raw/reactions/" + std::to_string(thisReactionCode);
                tempHistMap[histNameBuffer] = (TH1D*) tempHistMap[anaSample->GetName() + "/Raw/MC"]->Clone();
                tempHistMap[histNameBuffer]->Reset("ICESM");
                tempHistMap[histNameBuffer]->SetName(histNameBuffer.c_str());
                tempHistMap[histNameBuffer]->SetTitle(reactionNamesAndColors[thisReactionCode].first.c_str());
            }
        }



        // Fill the histograms (MC)
        for( size_t iEvent = 0 ; iEvent < anaSample->GetMcEvents().size() ; iEvent++ ){
            auto* anaEvent = &anaSample->GetMcEvents()[iEvent];
            tempHistMap[anaSample->GetName() + "/D1/MC"]->Fill(
                anaEvent->GetRecoD1(), anaEvent->GetEvWght()
            );
            tempHistMap[anaSample->GetName() + "/D2/MC"]->Fill(
                anaEvent->GetRecoD2(), anaEvent->GetEvWght()
            );
            tempHistMap[anaSample->GetName() + "/D1/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
                anaEvent->GetRecoD1(), anaEvent->GetEvWght()
            );
            tempHistMap[anaSample->GetName() + "/D2/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
                anaEvent->GetRecoD2(), anaEvent->GetEvWght()
            );
            tempHistMap[anaSample->GetName() + "/Raw/reactions/" + std::to_string(anaEvent->GetReaction())]->Fill(
                anaEvent->GetRecoBinIndex() + 0.5, anaEvent->GetEvWght()
            );
        }

        // Fill the histograms (Data)
        for( size_t iEvent = 0 ; iEvent < anaSample->GetDataEvents().size() ; iEvent++ ){
            auto* anaEvent = &anaSample->GetDataEvents()[iEvent];
            tempHistMap[anaSample->GetName() + "/D1/Data"]->Fill(
                anaEvent->GetRecoD1(), anaEvent->GetEvWght()
            );
            tempHistMap[anaSample->GetName() + "/D2/Data"]->Fill(
                anaEvent->GetRecoD2(), anaEvent->GetEvWght()
            );
        }

        // Cosmetics, Normalization and Write
        for(auto& histPair : tempHistMap){

            auto pathElements = GenericToolbox::splitString(histPair.first, "/");
            std::string xVarName = pathElements[1]; // SampleName/Var/SplitVar/...
            std::string splitVarName;
            std::string splitVarCode;
            if(pathElements.size() >= 4){
                splitVarName = pathElements[2]; // "reactions" for example
                splitVarCode = pathElements[3]; // "0" for example
            }

            histPair.second->GetXaxis()->SetTitle(xVarName.c_str());
            if(xVarName == "D1"){
                // Get Number of counts per 100 MeV
                for(int iBin = 0 ; iBin <= histPair.second->GetNbinsX()+1 ; iBin++){
                    histPair.second->SetBinContent( iBin, histPair.second->GetBinContent(iBin)/histPair.second->GetBinWidth(iBin)*100.);
                    histPair.second->SetBinError( iBin, TMath::Sqrt(histPair.second->GetBinContent(iBin))/histPair.second->GetBinWidth(iBin)*100.);
                }
                histPair.second->GetYaxis()->SetTitle("Counts/(100 MeV)");
                histPair.second->GetXaxis()->SetRangeUser(histPair.second->GetXaxis()->GetXmin(),maxD1Scale);
            }
            else {
                histPair.second->GetYaxis()->SetTitle("Counts");
            }

            if(pathElements.back() == "Data"){
                // IS DATA
                if( anaSample->GetDataType() == DataType::kAsimov
                    and xVarName != "Raw" // RAW IS NOT RENORMALIZED
                   ){
                    histPair.second->Scale(anaSample->GetNorm());
                }
                for( int iBin = 0 ; iBin <= histPair.second->GetNbinsX()+1 ; iBin++ ){
                    histPair.second->SetBinError(iBin, TMath::Sqrt(histPair.second->GetBinContent(iBin)));
                }
                histPair.second->SetLineColor(kBlack);
                histPair.second->SetMarkerColor(kBlack);
                histPair.second->SetMarkerStyle(kFullDotLarge);
                histPair.second->SetOption("EP");
            }
            else{
                // IS MC (if it's broken down by reaction, is MC too)
                if(xVarName != "Raw" or not splitVarName.empty()){ // RAW IS NOT RENORMALIZED, unless we rebuild it (for split var)
                    histPair.second->Scale(anaSample->GetNorm());
                }

                if(splitVarName == "reactions"){
                    histPair.second->SetLineColor(reactionNamesAndColors[stoi(splitVarCode)].second);
//                    histPair.second->SetFillColorAlpha(reactionNamesAndColors[stoi(splitVarCode)].second, 0.8);
                    histPair.second->SetFillColor(reactionNamesAndColors[stoi(splitVarCode)].second);
                    histPair.second->SetMarkerColor(reactionNamesAndColors[stoi(splitVarCode)].second);
                }
                else if( not splitVarName.empty() ){
                    histPair.second->SetLineColor(
                        colorWheel[splitVarColor[splitVarName]% colorWheel.size()]);
                    histPair.second->SetFillColor(
                        colorWheel[splitVarColor[splitVarName]% colorWheel.size()]);
                    histPair.second->SetMarkerColor(
                        colorWheel[splitVarColor[splitVarName]% colorWheel.size()]);
                    splitVarColor[splitVarName]++;
                }
                else{
                    histPair.second->SetLineColor(colorWheel[sampleColor% colorWheel.size()]);
                    histPair.second->SetFillColor(colorWheel[sampleColor% colorWheel.size()]);
                    histPair.second->SetMarkerColor(colorWheel[sampleColor% colorWheel.size()]);
                }

                histPair.second->SetOption("HIST");
            }

            histPair.second->SetLineWidth(2);
            histPair.second->GetYaxis()->SetRangeUser(
                histPair.second->GetMinimum(0), // 0 or lower as min will prevent to set log scale
                histPair.second->GetMaximum()*1.2
            );

            // Writing the histogram
            std::string subFolderPath = GenericToolbox::joinVectorString(pathElements, "/", 0, -1);
            GenericToolbox::mkdirTFile(samplesDir, subFolderPath)->cd();
            TH1D* tempHistPtr = (TH1D*) histPair.second->Clone();
            tempHistPtr->Write((pathElements.back() + "_TH1D").c_str());

        }

        GenericToolbox::appendToMap(TH1D_handler, tempHistMap);

        // Next Loop
        sampleColor++;

    }


    // Building canvas
    LogInfo << "Generating and writing sample canvas..." << std::endl;
    std::map<std::string, std::vector<TCanvas*>> canvasHandler;
    int nbXPlots       = 3;
    int nbYPlots       = 2;
    int nbSamples      = _samplesList_.size();


    int sampleCounter = 0;
    while(sampleCounter != nbSamples){
        std::stringstream canvasName;
        canvasName << "samples_" << sampleCounter+1;
        sampleCounter += nbXPlots*nbYPlots;
        if(sampleCounter > nbSamples){
            sampleCounter = nbSamples;
        }
        canvasName << "_to_" << sampleCounter;

        std::string pathBuffer;
        for(auto& canvasSubFolder : canvasSubFolderList){
            pathBuffer = canvasSubFolder + "/" + canvasName.str();
            canvasHandler[canvasSubFolder].emplace_back(new TCanvas(pathBuffer.c_str(), canvasName.str().c_str(), 1200, 700));
            canvasHandler[canvasSubFolder].back()->Divide(nbXPlots,nbYPlots);
        }
    }


    for(auto& canvasFolderPair : canvasHandler){

        int canvasIndex = 0;
        int iSlot       = 1;
        for(const auto& anaSample : _samplesList_){

            if(iSlot > nbXPlots*nbYPlots){
                canvasIndex++;
                iSlot = 1;
            }

            canvasFolderPair.second[canvasIndex]->cd(iSlot);

            auto subFolderList = GenericToolbox::splitString(canvasFolderPair.first, "/", true);
            std::string xVarName = subFolderList[0];

            TH1D* dataSampleHist = TH1D_handler[anaSample->GetName() + "/" + xVarName + "/Data"];
            std::vector<TH1D*> mcSampleHistList;

            if(subFolderList.size() == 1){ // no splitting of MC hist
                mcSampleHistList.emplace_back(TH1D_handler[anaSample->GetName() + "/" + xVarName + "/MC"]);
            }
            else{
                std::string splitVarName = subFolderList[1];
                for(auto& histNamePair : TH1D_handler){
                    if(GenericToolbox::doesStringStartsWithSubstring(
                        histNamePair.first,
                        anaSample->GetName() + "/" + xVarName + "/" + splitVarName + "/"
                    )){
                        mcSampleHistList.emplace_back(histNamePair.second);
                    }
                }
            }

            if(mcSampleHistList.empty()) continue;

            // Sorting histograms by norm (lowest stat first)
            std::function<bool(TH1D*, TH1D*)> aGoesFirst = [maxD1Scale](TH1D* histA_, TH1D* histB_){
                bool aGoesFirst = true; // A is smaller = A goes first
                double Xmax = histA_->GetXaxis()->GetXmax();
                if(maxD1Scale > 0) Xmax = maxD1Scale;
                if(  histA_->Integral(histA_->FindBin(0), histA_->FindBin(Xmax)-1)
                     > histB_->Integral(histB_->FindBin(0), histB_->FindBin(Xmax)-1) ) aGoesFirst = false;
                return aGoesFirst;
            };
            auto p = GenericToolbox::getSortPermutation( mcSampleHistList, aGoesFirst );
            mcSampleHistList = GenericToolbox::applyPermutation(mcSampleHistList, p);

            // Stacking histograms
            TH1D* histPileBuffer = nullptr;
            double minYValue = 1;
            for( size_t iHist = 0 ; iHist < mcSampleHistList.size() ; iHist++ ){
                if(minYValue > mcSampleHistList.back()->GetMinimum(0)){
                    minYValue = mcSampleHistList.back()->GetMinimum(0);
                }
                if(histPileBuffer != nullptr) mcSampleHistList[iHist]->Add(histPileBuffer);
                histPileBuffer = mcSampleHistList[iHist];
            }

            // Draw (the one on top of the pile should be drawn first, otherwise it will hide the others...)
            int lastIndex = mcSampleHistList.size()-1;
            for( int iHist = lastIndex ; iHist >= 0 ; iHist-- ){
                if( iHist == lastIndex ){
                    mcSampleHistList[iHist]->GetYaxis()->SetRangeUser(
                        minYValue,
                        mcSampleHistList[iHist]->GetMaximum()*1.2
                        );
                    mcSampleHistList[iHist]->Draw("HIST");
                }
                else {
                    mcSampleHistList[iHist]->Draw("HISTSAME");
                }
            }

            dataSampleHist->SetTitle("Data");
            dataSampleHist->Draw("EPSAME");

            // Legend
            double Xmax = 0.9;
            double Ymax = 0.9;
            double Xmin = 0.5;
            double Ymin = Ymax - 0.04*(mcSampleHistList.size()+1);
            gPad->BuildLegend(Xmin,Ymin,Xmax,Ymax);

            mcSampleHistList[lastIndex]->SetTitle(anaSample->GetName().c_str()); // the actual title
            gPad->SetGridx();
            gPad->SetGridy();
            iSlot++;

        } // sample

        samplesDir->cd();
        GenericToolbox::mkdirTFile(samplesDir, "canvas/" + canvasFolderPair.first)->cd();
        for(auto& canvas : canvasFolderPair.second){
            canvas->Write((canvas->GetTitle() + std::string("_TCanvas")).c_str());
        }

    } // canvas



}


void ND280Fitter::InitializeThreadsParameters(){

    if(GlobalVariables::getNbThreads() > 1){

        LogWarning << "Initializing threads workers..." << std::endl;

        std::function<void(int)> asyncLoop = [this](int iThread_){
            while(not _stopThreads_){
                // Pending state loop
//                std::this_thread::sleep_for(std::chrono::microseconds(33)); // 30,000 fps cap

                if(_triggerReweightThreads_.at(iThread_)){
                    this->ReWeightEvents(iThread_);
                    _triggerReweightThreads_.at(iThread_) = false; // toggle off the trigger
                }

                if(_triggerReFillMcHistogramsThreads_.at(iThread_)){
                    for( size_t iSample = 0 ; iSample < _samplesList_.size() ; iSample++ ){
                        _samplesList_[iSample]->FillMcHistograms(iThread_);
                    }
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
        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads()-1 ; iThread++ ){
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

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in _fitParameterIsFixedList_:
        std::vector<bool> anaParFixedList;
        _fitParametersGroupList_[iGroup]->GetParFixed(anaParFixedList);
        _fitParameterIsFixedList_.insert(_fitParameterIsFixedList_.end(), anaParFixedList.begin(), anaParFixedList.end());
    }

    // Nothing to fit with zero fit parameters:
    if(_nb_fit_parameters_ == 0){
        LogError << "No fit parameters were defined." << std::endl;
        return;
    }

    LogWarning << "Propagating prior systematics..." << std::endl;
    this->PropagateSystematics();

}
void ND280Fitter::InitializeDataSamples(){

    LogWarning << "Initializing data samples..." << std::endl;

    if(_selected_data_type_ == kReset or _selected_data_type_ == kMC){
        LogFatal << "In " << __METHOD_NAME__ << std::endl;
        LogFatal << "No valid _selected_data_type_ provided." << std::endl;
        throw std::logic_error("No valid _selected_data_type_ provided.");
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

        if(_fitParameterIsFixedList_[iPar]){
            _minimizer_->FixVariable(iPar);
        }
    }

    LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
            << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
            << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
            << std::endl;

}

void ND280Fitter::PassIfInitialized(const std::string& method_name_) const{
    if(not _isInitialized_){
        LogFatal << "Can't do " << method_name_ << " while not initialized." << std::endl;
        throw std::logic_error("Can't do " + method_name_ + " while not initialized.");
    }
}
void ND280Fitter::PassIfNotInitialized(const std::string& method_name_) const{
    if(_isInitialized_){
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

double ND280Fitter::EvalFit(const double* par){

    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(FIT_IT_TIME_POINT);

    _printFitState_ = false;

    _nbFitCalls_++;
    if( _fitHasStarted_ and (_nbFitCalls_ < 20 or _nbFitCalls_ % 100 == 0) ){
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

    if(_nbFitCalls_ % _saveFitParamsFrequency_ == 0 and _saveFitParameters_){
        SaveParams(_newParametersList_);
        SaveEventHist(_nbFitCalls_);
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
    for(int iGroup = 0; iGroup < _fitParametersGroupList_.size(); iGroup++){
        for(int iParameter = 0; iParameter < _fitParametersGroupList_[iGroup]->GetNpar(); iParameter++){
            _newParametersList_[iGroup][iParameter] = par[iMinuitParIndex];
            iMinuitParIndex++;
        }

        if(_fitParametersGroupList_[iGroup]->IsDecomposed()){
            _newParametersList_[iGroup] = _fitParametersGroupList_[iGroup]->GetOriginalParameters(
                    _newParametersList_[iGroup]
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

    if( GlobalVariables::getNbThreads() > 1
       and not _disableMultiThread_
//       and false // DISABLE FOR DEBUG
       ){

        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads()-1; iThread++){
            // Start every waiting thread
            _triggerReweightThreads_[iThread] = true;
        }

        this->ReWeightEvents(GlobalVariables::getNbThreads()-1); // last one performed by this thread

        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads()-1; iThread++){
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
        if(_advancedTimeMonitoring_){
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

    if( GlobalVariables::getNbThreads() > 1
//        and false // DISABLE FOR DEBUG
        and not _disableMultiThread_
       ){
        // Starting refill threads
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads()-1 ; iThread++){
            _triggerReFillMcHistogramsThreads_[iThread] = true;
        }

        for( size_t iSample = 0 ; iSample < _samplesList_.size() ; iSample++ ){
            _samplesList_[iSample]->FillMcHistograms(GlobalVariables::getNbThreads()-1);
        }

        // Wait for threads to finish their jobs
        for(int iThread = 0 ; iThread < GlobalVariables::getNbThreads()-1 ; iThread++){
            while(_triggerReFillMcHistogramsThreads_[iThread]){
                // triggerReweight are set to false when the thread is finished
//                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }

        // MERGE HISTOGRAMS
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // make sure everybody has finished...
        for( size_t iSample = 0 ; iSample < _samplesList_.size() ; iSample++ ){
            _samplesList_[iSample]->MergeMcHistogramsThread();
        }
    } // _nbThreads_ > 1
    else {
        // single core
        for( size_t iSample = 0 ; iSample < _samplesList_.size() ; iSample++ ){
            _samplesList_[iSample]->FillMcHistograms();
        }
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
    _chi2StatBuffer_ = 0; // reset
    double buffer;
    for(auto & sampleContainer : _samplesList_){

        //buffer = _samplesList_.at(sampleContainer)->CalcChi2();
        buffer = sampleContainer->CalcLLH();
        //buffer = _samplesList_.at(sampleContainer)->CalcEffLLH();

        if(_printFitState_){
            LogInfo << "Chi2 stat for sample " << sampleContainer->GetName() << " is "
                    << buffer << std::endl;
        }

        _chi2StatBuffer_ += buffer;

    }


    ////////////////////////////////
    // Compute the penalty terms
    ////////////////////////////////
    _chi2PullsBuffer_ = 0;
    _chi2RegBuffer_ = 0;
    for(size_t iGroup = 0; iGroup < _fitParametersGroupList_.size(); iGroup++){

        if(not _disableChi2Pulls_){
            buffer = _fitParametersGroupList_[iGroup]->GetChi2(_newParametersList_[iGroup]);
            _chi2PullsBuffer_ += buffer;
            if(_printFitState_){
                LogInfo << "Chi2 contribution from " << _fitParametersGroupList_[iGroup]->GetName() << " is "
                        << buffer
                        << std::endl;
            }
        }

        if(_fitParametersGroupList_[iGroup]->IsRegularised()){
            _chi2RegBuffer_ += _fitParametersGroupList_[iGroup]->CalcRegularisation(_newParametersList_[iGroup]);
        }

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

}

// Multi-thread compatible methods
void ND280Fitter::ReWeightEvents(int iThread_){

    AnaEvent* currentEventPtr;
    AnaSample* currentSamplePtr;
    bool isMultiThreaded = (iThread_ != -1);
    if(not isMultiThreaded) iThread_ = 0;
    size_t nEvents;

    if(_advancedTimeMonitoring_){
        GlobalVariables::getThreadMutex().lock();
        GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+(iThread_+1)); // Claim memory (avoid seg fault)
        GlobalVariables::getThreadMutex().unlock();
    }

    // Looping over all events
    for( size_t iSample = 0; iSample < _samplesList_.size(); iSample++ ){
        currentSamplePtr = _samplesList_.at(iSample);
        nEvents = currentSamplePtr->GetNbMcEvents();
        for( size_t iEvent = 0; iEvent < nEvents ; iEvent++ ){

            if(isMultiThreaded){
                if( iEvent % GlobalVariables::getNbThreads() != iThread_ ){
                    continue; // skip this event
                }
            }

            currentEventPtr = &currentSamplePtr->GetMcEvents().at(iEvent);

            // lock (even if it's useless: for debug)
            while(currentEventPtr->GetIsBeingEdited()){
                // THIS SHOULD NOT HAPPENED
                LogAlert << "WAITING (more than 1 thread editing this event?)...." << std::endl;
            }
            currentEventPtr->SetIsBeingEdited(true);

            currentEventPtr->ResetEvWght();

            for(size_t jGroup = 0 ; jGroup < _fitParametersGroupList_.size() ; jGroup++){
                if(_advancedTimeMonitoring_){
                    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+(iThread_+1));
                }
                _fitParametersGroupList_.at(jGroup)->ReWeight(
                    currentEventPtr, currentSamplePtr->GetDetector(),
                    iSample, iEvent,_newParametersList_.at(jGroup)
                );
                if(_advancedTimeMonitoring_){
                    _durationReweightParameters_[jGroup][iThread_].back() += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(10000+(iThread_+1));
                }
            } // jParam

            if(currentEventPtr->GetEvWght() < 0){
//                GlobalVariables::getThreadMutex().lock();
//                LogError <<  "Event#" << currentEventPtr->GetEvId() << ": " << GET_VAR_NAME_VALUE(currentEventPtr->GetEvWght()) << std::endl;
//                throw std::runtime_error("Event has a negative weight.");
                currentEventPtr->SetEvWght(0);
            }

            // unlock
            currentEventPtr->SetIsBeingEdited(false);

        } // iEvent

    } // iSample

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

