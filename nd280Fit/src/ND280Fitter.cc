#include "ND280Fitter.hh"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"
#include "Logger.h"
#include <TCanvas.h>
#include <THStack.h>
#include <iostream>

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
    _fit_has_been_called_ = false;
    _fit_has_converged_ = false;

    _PRNG_seed_ = -1;

    delete _minimizer_;         _minimizer_ = nullptr;
    delete _functor_;           _functor_ = nullptr;

    _output_tdirectory_ = nullptr; // not handle by this class

    _save_fit_params_   = false;
    _save_event_tree_   = true;
    _disable_syst_fit_  = false;
    _apply_statistical_fluctuations_on_samples_ = false;

    _save_fit_params_frequency_  = 10000;
    _nb_threads_                 = 1;
    _nb_fit_parameters_          = 0;
    _nb_fit_calls_               = 0;

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

    _AnaFitParameters_list_.clear();
    _AnaSample_list_.clear();

    _parameter_names_.clear();
    _parameter_prefit_values_.clear();

}

void ND280Fitter::SetOutputTDirectory(TDirectory* output_tdirectory_) {
    _output_tdirectory_ = output_tdirectory_;
}
void ND280Fitter::SetPrngSeed(int PRNG_seed_){
    _PRNG_seed_ = PRNG_seed_;
}
void ND280Fitter::SetNbThreads(int nb_threads_){
    _nb_threads_ = nb_threads_;
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
    _disable_syst_fit_ = disable_syst_fit_;
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
    _save_fit_params_frequency_ = save_fit_params_frequency_;
}
void ND280Fitter::SetAnaFitParametersList(std::vector<AnaFitParameters*> AnaFitParameters_list_){
    _AnaFitParameters_list_ = std::move(AnaFitParameters_list_);
}
void ND280Fitter::SetAnaSamplesList(std::vector<AnaSample*> AnaSample_list_){
    _AnaSample_list_ = AnaSample_list_;
}
void ND280Fitter::SetSelectedDataType(int selected_data_type_){
    _selected_data_type_ = static_cast<DataType>(selected_data_type_);
}


void ND280Fitter::Initialize(){

    LogWarning << "ND280 Fitter is initializing..." << std::endl;
    PassIfNotInitialized(__METHOD_NAME__);

    if(_output_tdirectory_ == nullptr){
        LogFatal << "_output_tdirectory_ has not been set." << std::endl;
        throw std::logic_error("_output_tdirectory_ has not been set.");
    }
    else if(_PRNG_seed_ == -1){
        LogFatal << "_PRNG_seed_ has not been set." << std::endl;
        throw std::logic_error("_PRNG_seed_ has not been set.");
    }
    else if(_AnaSample_list_.empty()){
        LogFatal << "_AnaSample_list_ has not been set." << std::endl;
        throw std::logic_error("_AnaSample_list_ has not been set.");
    }
    else if(_selected_data_type_ == kReset){
        LogFatal << "_selected_data_type_ has not been set." << std::endl;
        throw std::logic_error("_selected_data_type_ has not been set.");
    }

    InitializeFitParameters();
    InitializeSamples();
    InitializeFitter();

    _is_initialized_ = true;
    LogWarning << "ND280 Fitter has been initialized." << std::endl;

}


void ND280Fitter::WritePrefitData(){

    LogInfo << "Writing prefit data..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    auto* cwd = _output_tdirectory_->mkdir("prefit");

    LogDebug << "Writing prefit histograms data..." << std::endl;
    TH1D h_prefit("hist_prefit_par_all",
                  "hist_prefit_par_all",
                  _nb_fit_parameters_,
                  0,
                  _nb_fit_parameters_);
    TVectorD v_prefit_original(_nb_fit_parameters_);
    TVectorD v_prefit_decomp(_nb_fit_parameters_);
    TVectorD v_prefit_start(_nb_fit_parameters_, _parameter_prefit_values_.data());

    int num_par = 1;
    for(const auto & anaFitParameters : _AnaFitParameters_list_){

        const unsigned int npar = anaFitParameters->GetNpar();

        TMatrixDSym* cov_mat = anaFitParameters->GetCovMat();
        for(int j = 0; j < anaFitParameters->GetNpar(); ++j){
            h_prefit.SetBinContent(num_par, anaFitParameters->GetParPrior(j));
            if(anaFitParameters->HasCovMat()){
                h_prefit.SetBinError(num_par, std::sqrt((*cov_mat)[j][j]));
            }
            else{
                h_prefit.SetBinError(num_par, 0);
            }

            v_prefit_original[num_par-1] = anaFitParameters->GetParOriginal(j);
            v_prefit_decomp[num_par-1] = anaFitParameters->GetParPrior(j);
            num_par++;
        }
    }

    cwd->cd();
    h_prefit.Write();
    v_prefit_original.Write("vec_prefit_original");
    v_prefit_decomp.Write("vec_prefit_decomp");
    v_prefit_start.Write("vec_prefit_start");

    SaveEventHist(0);


    LogDebug << "Writing prefit samples plots..." << std::endl;
    std::vector<Color_t> colorCells = {kGreen-3, kTeal+3, kAzure+7, kCyan-2, kBlue-7, kBlue+2, kOrange+1, kOrange+9, kRed+2, kPink+9};

    std::map<std::string, TH1D*> TH1D_handler;
    int iColor = 0;
    auto* samplesDir = cwd->mkdir("samples");
    for(const auto& anaSample : _AnaSample_list_ ){

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
        TH1D_handler[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
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
            TH1D_handler[tempHistName] = new TH1D(tempHistName.c_str(), tempHistName.c_str(),
                                                  D1binning.size() - 1, &D1binning[0]);
        }


        // Fill the histograms
        for(int iEvent = 0 ; iEvent < anaSample->GetN() ; iEvent++){
            auto* anaEvent = anaSample->GetEvent(iEvent);
            TH1D_handler[anaSample->GetName() + "/D1"]->Fill(anaEvent->GetRecoD1()
//                                                                 , anaEvent->GetEvWght()*anaSample->GetNorm()
                                                             );
            TH1D_handler[anaSample->GetName() + "/reactions/D1_reaction_"+std::to_string(anaEvent->GetReaction())]->Fill(
                anaEvent->GetRecoD1()
//                    , anaEvent->GetEvWght()
                );
        }

        // Next Loop
        iColor++;
        samplesDir->cd();
    }

    // Renormalize each bin to its width
    for(auto& hist : TH1D_handler){
        for(int iBin = 0 ; iBin < hist.second->GetNbinsX() ; iBin++){
            hist.second->SetBinContent(iBin,hist.second->GetBinContent(iBin)/hist.second->GetBinWidth(iBin)*100.);
            hist.second->SetBinError(iBin,hist.second->GetBinError(iBin)/hist.second->GetBinWidth(iBin)*100.);
        }
        for(const auto& anaSample : _AnaSample_list_ ){
            if(GenericToolbox::doesStringStartsWithSubstring(hist.first, anaSample->GetName())){
                hist.second->Scale(anaSample->GetNorm());
                break;
            }
        }
    }

    // Makeup
    for(auto& hist : TH1D_handler){

        hist.second->SetLineColor(colorCells[iColor]);
        hist.second->SetLineWidth(2);
        hist.second->SetMarkerColor(colorCells[iColor]);
        if(GenericToolbox::doesStringContainsSubstring(hist.first, "D1")){
            hist.second->GetXaxis()->SetTitle("D1 Reco");
        }
        hist.second->GetYaxis()->SetTitle("Counts/(100 MeV)");
        hist.second->GetYaxis()->SetRangeUser(0, hist.second->GetMaximum()*1.2);

    }

    for(auto& hist : TH1D_handler){
        // Looking for a subfolder in the hist name
        auto nameSlices = GenericToolbox::splitString(hist.second->GetName(), "/");
        std::string histName = nameSlices[nameSlices.size()-1];
        std::string histSubFolder = GenericToolbox::joinVectorString(nameSlices, "/", 0, -1);
        if(samplesDir->GetDirectory(histSubFolder.c_str()) == nullptr) samplesDir->mkdir(histSubFolder.c_str());
        samplesDir->cd(histSubFolder.c_str());

        // Writing histo
        hist.second->Write(histName.c_str());

        // Coming back to samples/ dir
        samplesDir->cd();
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
    for(const auto& anaSample : _AnaSample_list_ ){
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
        histToPlotList[0]->GetXaxis()->SetRangeUser(0,lastBinLowEdge);
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

    for(auto& canvas : canvasHandler){
        canvas.second->Write();
    }


    // Systematics
    LogDebug << "Writing prefit systematics data..." << std::endl;
    TMatrixD* tempTMatrixD;
    TH2D* tempTH2D;
    auto* systematicsDir = cwd->mkdir("systematics");
    systematicsDir->cd();
    for(auto & AnaFitParameters : _AnaFitParameters_list_){
        systematicsDir->mkdir(AnaFitParameters->GetName().c_str());
        systematicsDir->cd(AnaFitParameters->GetName().c_str());

        tempTMatrixD = (TMatrixD*) AnaFitParameters->GetCovMat();
        tempTMatrixD->Write((AnaFitParameters->GetName()+"_Cov_TMatrixD").c_str());

        tempTH2D = GenericToolbox::convertTMatrixDtoTH2D(tempTMatrixD,AnaFitParameters->GetName());
        std::vector<std::string> parNameList;
        AnaFitParameters->GetParNames(parNameList);

        for( int iBin = 1 ; iBin < tempTH2D->GetXaxis()->GetNbins() ; iBin++ ){
            tempTH2D->GetXaxis()->SetBinLabel(iBin, parNameList[iBin-1].c_str());
            tempTH2D->GetYaxis()->SetBinLabel(iBin, parNameList[iBin-1].c_str());
        }
        tempTH2D->SetTitle((AnaFitParameters->GetName() + " Covariance Matrix").c_str());
        tempTH2D->Write((AnaFitParameters->GetName() + "_Cov_TH2D").c_str());

        tempTMatrixD = GenericToolbox::convertToCorrelationMatrix(tempTMatrixD);
        tempTMatrixD->Write((AnaFitParameters->GetName()+"_Cor_TMatrixD").c_str());
        tempTH2D = GenericToolbox::convertTMatrixDtoTH2D(tempTMatrixD,AnaFitParameters->GetName());
        for( int iBin = 1 ; iBin < tempTH2D->GetXaxis()->GetNbins() ; iBin++ ){
            tempTH2D->GetXaxis()->SetBinLabel(iBin, parNameList[iBin-1].c_str());
            tempTH2D->GetYaxis()->SetBinLabel(iBin, parNameList[iBin-1].c_str());
        }
        tempTH2D->SetTitle((AnaFitParameters->GetName() + " Correlation Matrix").c_str());
        tempTH2D->Write((AnaFitParameters->GetName() + "_Cor_TH2D").c_str());

        systematicsDir->cd();
    }

    tempTMatrixD = GeneratePriorCovarianceMatrix();
    tempTMatrixD->Write("PriorCovarianceMatrix_TMatrixD");
    tempTH2D = GenericToolbox::convertTMatrixDtoTH2D(tempTMatrixD,"PriorCovarianceMatrix");
    tempTH2D->Write("PriorCovarianceMatrix_TH2D");

    _output_tdirectory_->cd();
}
bool ND280Fitter::Fit(){

    LogAlert << "Starting to fit." << std::endl;
    PassIfInitialized(__METHOD_NAME__);
    _fit_has_been_called_ = true;

    LogInfo << "Calling Minimize, running: " << _minimization_settings_.algorithm << std::endl;

    // Run the actual fitter:
    _fit_has_converged_ = _minimizer_->Minimize();

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

    return _fit_has_converged_;

}
void ND280Fitter::WritePostFitData(){

    LogInfo << "Writing postfit data..." << std::endl;
    PassIfInitialized(__METHOD_NAME__);

    if(not _fit_has_been_called_){
        LogFatal << "Can't write post fit data: fit has not been called previously" << std::endl;
        throw std::logic_error("Can't write post fit data: fit has not been called previously");
    }

    auto* cwd = _output_tdirectory_->mkdir("postfit");
    cwd->cd();

    // Saving Chi2
    TH1D h_chi2stat("chi2_stat_periter", "chi2_stat_periter", _nb_fit_calls_ + 1, 0,
                    _nb_fit_calls_ + 1);
    TH1D h_chi2sys("chi2_sys_periter", "chi2_sys_periter", _nb_fit_calls_ + 1, 0,
                   _nb_fit_calls_ + 1);
    TH1D h_chi2reg("chi2_reg_periter", "chi2_reg_periter", _nb_fit_calls_ + 1, 0,
                   _nb_fit_calls_ + 1);
    TH1D h_chi2tot("chi2_tot_periter", "chi2_tot_periter", _nb_fit_calls_ + 1, 0,
                   _nb_fit_calls_ + 1);

    if(vec_chi2_stat.size() != vec_chi2_sys.size())
    {
        std::cout  << "Number of saved iterations for chi2 stat and chi2 syst are different."
                   << std::endl;
    }
    for(size_t i = 0; i < vec_chi2_stat.size(); i++)
    {
        h_chi2stat.SetBinContent(i + 1, vec_chi2_stat[i]);
        h_chi2sys.SetBinContent(i + 1, vec_chi2_sys[i]);
        h_chi2reg.SetBinContent(i + 1, vec_chi2_reg[i]);
        h_chi2tot.SetBinContent(i + 1, vec_chi2_sys[i] + vec_chi2_stat[i] + vec_chi2_reg[i]);
    }

    _output_tdirectory_->cd();
    h_chi2stat.Write();
    h_chi2sys.Write();
    h_chi2reg.Write();
    h_chi2tot.Write();

    const int ndim        = _minimizer_->NDim();
    const int nfree       = _minimizer_->NFree();
    const double* par_val = _minimizer_->X();
    const double* par_err = _minimizer_->Errors();
    double cov_array[ndim * ndim];
    _minimizer_->GetCovMatrix(cov_array);

    std::vector<double> par_val_vec(par_val, par_val + ndim);
    std::vector<double> par_err_vec(par_err, par_err + ndim);

    unsigned int par_offset = 0;
    TMatrixDSym cov_matrix(ndim, cov_array);
    for(const auto& fit_param : _AnaFitParameters_list_){
        if(fit_param->IsDecomposed()){
            cov_matrix  = fit_param->GetOriginalCovMat(cov_matrix, par_offset);
            par_val_vec = fit_param->GetOriginalParameters(par_val_vec, par_offset);
        }
        par_offset += fit_param->GetNpar();
    }

    TMatrixDSym cor_matrix(ndim);
    for(int r = 0; r < ndim; ++r)
    {
        for(int c = 0; c < ndim; ++c)
        {
            cor_matrix[r][c] = cov_matrix[r][c] / std::sqrt(cov_matrix[r][r] * cov_matrix[c][c]);
            if(std::isnan(cor_matrix[r][c]))
                cor_matrix[r][c] = 0;
        }
    }

    TVectorD postfit_globalcc(ndim);
    for(int i = 0; i < ndim; ++i)
        postfit_globalcc[i] = _minimizer_->GlobalCC(i);

    TVectorD postfit_param(ndim, &par_val_vec[0]);
    std::vector<std::vector<double>> res_pars;
    std::vector<std::vector<double>> err_pars;
    int k = 0;
    for(int i = 0; i < _AnaFitParameters_list_.size(); i++)
    {
        const unsigned int npar = _AnaFitParameters_list_[i]->GetNpar();
        std::vector<double> vec_res;
        std::vector<double> vec_err;

        for(int j = 0; j < npar; j++)
        {
            vec_res.emplace_back(par_val_vec[k]);
            vec_err.emplace_back(std::sqrt(cov_matrix[k][k]));
            k++;
        }

        res_pars.emplace_back(vec_res);
        err_pars.emplace_back(vec_err);
    }

    cwd->cd();
    cov_matrix.Write("res_cov_matrix");
    cor_matrix.Write("res_cor_matrix");
    postfit_param.Write("res_vector");
    postfit_globalcc.Write("res_globalcc");

    SaveResults(res_pars, err_pars);
    SaveEventHist(_nb_fit_calls_, true);

    if(_save_event_tree_) SaveEventTree(res_pars);

}


void ND280Fitter::InitializeFitParameters(){

    LogWarning << "Initializing the fit parameters..." << std::endl;

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    std::vector<std::vector<double>> new_pars;
    for(const auto & anaFitParameters : _AnaFitParameters_list_) {
        // _nb_fit_parameters_ is the number of total fit paramters:
        _nb_fit_parameters_ += anaFitParameters->GetNpar();

        // Get names of all the different parameters (for all parameter types) and store them in par_names:
        std::vector<std::string> ana_par_names;
        anaFitParameters->GetParNames(ana_par_names);
        _parameter_names_.insert(_parameter_names_.end(), ana_par_names.begin(),ana_par_names.end());

        // Get the priors for this parameter type (should be 1 unless decomp has been set to true in the .json config file) and store them in vec1:
        std::vector<double> anaParPriors;
        anaFitParameters->GetParPriors(anaParPriors);

        // If rng_template has been set to true in the .json config file, the template parameters will be randomized (around 1) according to a gaussian distribution:
        if(anaFitParameters->DoRNGstart())
        {
            LogInfo << "Randomizing start point for " << anaFitParameters->GetName() << std::endl;
            for(auto& p : anaParPriors)
                p += (p * _PRNG_->Gaus(0.0, 0.1));
        }

        // Store the prefit values (for all parameter types) in par_prefit:
        _parameter_prefit_values_.insert(_parameter_prefit_values_.end(), anaParPriors.begin(), anaParPriors.end());
        new_pars.emplace_back(anaParPriors);

        // Store the pars_step values (for all parameter types) and store them in _parameter_steps_:
        anaFitParameters->GetParSteps(anaParPriors);
        _parameter_steps_.insert(_parameter_steps_.end(), anaParPriors.begin(), anaParPriors.end());

        // Store the lower and upper limits for the fit parameters (for all parameter types) in _parameter_low_edges_ and _parameter_high_edges_:
        std::vector<double> anaParLowEdges, anaParHighEdges;
        anaFitParameters->GetParLimits(anaParLowEdges, anaParHighEdges);
        _parameter_low_edges_.insert( _parameter_low_edges_.end(), anaParLowEdges.begin(), anaParLowEdges.end() );
        _parameter_high_edges_.insert( _parameter_high_edges_.end(), anaParHighEdges.begin(), anaParHighEdges.end() );

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in _parameter_fixed_list_:
        std::vector<bool> anaParFixedList;
        anaFitParameters->GetParFixed(anaParFixedList);
        _parameter_fixed_list_.insert(_parameter_fixed_list_.end(), anaParFixedList.begin(), anaParFixedList.end());
    }

    // Nothing to fit with zero fit parameters:
    if(_nb_fit_parameters_ == 0)  {
        LogError << "No fit parameters were defined." << std::endl;
        return;
    }

    LogWarning << "Applying prior event reweight (filling spline cache)..." << std::endl;
    for( int iSample = 0 ; iSample < int(_AnaSample_list_.size()) ; iSample++ ){
        LogInfo << "Processing " << _AnaSample_list_[iSample]->GetName() << std::endl;
        std::string detectorName = _AnaSample_list_[iSample]->GetDetector();
        for(int iEvent = 0 ; iEvent < _AnaSample_list_[iSample]->GetN() ; iEvent++){
            GenericToolbox::displayProgressBar(iEvent, _AnaSample_list_[iSample]->GetN(), LogInfo.getPrefixString() + "Reweighting and filling prefit histograms...");
            auto* anaEvent = _AnaSample_list_[iSample]->GetEvent(iEvent);
            anaEvent->ResetEvWght();
            // Loop over all the different parameter types such as [template, flux, detector, cross section]:
            for(int jParameter = 0; jParameter < _AnaFitParameters_list_.size(); jParameter++) {
                // Multiply the current event weight for event ev with the paramter of the current parameter type for the (truth bin)/(reco bin)/(energy bin) that this event falls in:
                _AnaFitParameters_list_[jParameter]->ReWeight(anaEvent, detectorName, iSample, iEvent, new_pars[jParameter]);
            }
        }
    }

}
void ND280Fitter::InitializeSamples(){

    if(_selected_data_type_ == kReset){
        LogFatal << "In " << __METHOD_NAME__ << std::endl;
        LogFatal << "No valid _selected_data_type_ provided." << std::endl;
        throw std::logic_error("No valid _selected_data_type_ provided.");
    }
    else if(_selected_data_type_ == kExternal){
        GenerateToyData(0, _apply_statistical_fluctuations_on_samples_);
    }
    else{
        for(const auto& anaSample : _AnaSample_list_){
            anaSample->SetNorm(1);
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

    // The ROOT Functor class is used to wrap multi-dimensional function objects, in this case the ND280Fitter::CalcLikelihood function calculates and returns chi2_stat + chi2_sys + chi2_reg in each iteration of the fitter:
    _functor_ = new ROOT::Math::Functor(this, &ND280Fitter::CalcLikelihood, _nb_fit_parameters_);

    _minimizer_->SetFunction(*_functor_);
    _minimizer_->SetStrategy(_minimization_settings_.strategy);
    _minimizer_->SetPrintLevel(_minimization_settings_.print_level);
    _minimizer_->SetTolerance(_minimization_settings_.tolerance);
    _minimizer_->SetMaxIterations(_minimization_settings_.max_iter);
    _minimizer_->SetMaxFunctionCalls(_minimization_settings_.max_fcn);

    for(int iPar = 0; iPar < _nb_fit_parameters_; iPar++){
        _minimizer_->SetVariable(
            iPar,_parameter_names_[iPar],
            _parameter_prefit_values_[iPar],
            _parameter_steps_[iPar]
        );

        if(_parameter_fixed_list_[iPar]){
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
    _AnaFitParameters_list_ = fitpara;

    // Vectors holding the settings for the fit parameters (for all parameter types):
    std::vector<double> par_step, par_low, par_high;
    std::vector<bool> par_fixed;

    // ROOT random number interface (seed of 0 means that seed is automatically computed based on time):
    //TRandom3 rng(0);

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(std::size_t i = 0; i < _AnaFitParameters_list_.size(); i++)
    {
        // m_npar is the number of total fit paramters:
        _nb_fit_parameters_ += _AnaFitParameters_list_[i]->GetNpar();

        // Get names of all the different parameters (for all parameter types) and store them in par_names:
        std::vector<std::string> vec0;
        _AnaFitParameters_list_[i]->GetParNames(vec0);
        _parameter_names_.insert(_parameter_names_.end(), vec0.begin(), vec0.end());

        // Get the priors for this parameter type (should be 1 unless decomp has been set to true in the .json config file) and store them in vec1:
        std::vector<double> vec1, vec2;
        _AnaFitParameters_list_[i]->GetParPriors(vec1);

        // If rng_template has been set to true in the .json config file, the template parameters will be randomized (around 1) according to a gaussian distribution:
        if(_AnaFitParameters_list_[i]->DoRNGstart())
        {
            LogInfo << "Randomizing start point for " << _AnaFitParameters_list_[i]->GetName() << std::endl;
            for(auto& p : vec1)
                p += (p * _PRNG_->Gaus(0.0, 0.1));
        }

        // Store the prefit values (for all parameter types) in par_prefit:
        _parameter_prefit_values_.insert(_parameter_prefit_values_.end(), vec1.begin(), vec1.end());

        // Store the pars_step values (for all parameter types) and store them in _parameter_steps_:
        _AnaFitParameters_list_[i]->GetParSteps(vec1);
        par_step.insert(par_step.end(), vec1.begin(), vec1.end());

        // Store the lower and upper limits for the fit parameters (for all parameter types) in _parameter_low_edges_ and _parameter_high_edges_:
        _AnaFitParameters_list_[i]->GetParLimits(vec1, vec2);
        par_low.insert(par_low.end(), vec1.begin(), vec1.end());
        par_high.insert(par_high.end(), vec2.begin(), vec2.end());

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in _parameter_fixed_list_:
        std::vector<bool> vec3;
        _AnaFitParameters_list_[i]->GetParFixed(vec3);
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

    // The ROOT Functor class is used to wrap multi-dimensional function objects, in this case the ND280Fitter::CalcLikelihood function calculates and returns chi2_stat + chi2_sys + chi2_reg in each iteration of the fitter:
    _functor_ = new ROOT::Math::Functor(this, &ND280Fitter::CalcLikelihood, _nb_fit_parameters_);

    _minimizer_->SetFunction(*_functor_);
    _minimizer_->SetStrategy(_minimization_settings_.strategy);
    _minimizer_->SetPrintLevel(_minimization_settings_.print_level);
    _minimizer_->SetTolerance(_minimization_settings_.tolerance);
    _minimizer_->SetMaxIterations(_minimization_settings_.max_iter);
    _minimizer_->SetMaxFunctionCalls(_minimization_settings_.max_fcn);

    for(int i = 0; i < _nb_fit_parameters_; ++i)
    {
        _minimizer_->SetVariable(i, _parameter_names_[i], _parameter_prefit_values_[i], par_step[i]);
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
    TVectorD v_prefit_start(_nb_fit_parameters_, _parameter_prefit_values_.data());

    int num_par = 1;
    for(int i = 0; i < _AnaFitParameters_list_.size(); ++i)
    {
        TMatrixDSym* cov_mat = _AnaFitParameters_list_[i]->GetCovMat();
        for(int j = 0; j < _AnaFitParameters_list_[i]->GetNpar(); ++j)
        {
            h_prefit.SetBinContent(num_par, _AnaFitParameters_list_[i]->GetParPrior(j));
            if(_AnaFitParameters_list_[i]->HasCovMat())
                h_prefit.SetBinError(num_par, std::sqrt((*cov_mat)[j][j]));
            else
                h_prefit.SetBinError(num_par, 0);

            v_prefit_original[num_par-1] = _AnaFitParameters_list_[i]->GetParOriginal(j);
            v_prefit_decomp[num_par-1] = _AnaFitParameters_list_[i]->GetParPrior(j);
            num_par++;
        }
    }

    _output_tdirectory_->cd();
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
    for(const auto& fitpar : _AnaFitParameters_list_)
    {
        std::vector<double> toy_throw(fitpar->GetNpar(), 0.0);
        fitpar -> ThrowPar(toy_throw, temp_seed++);

        chi2_syst += fitpar -> GetChi2(toy_throw);
        fitpar_throw.emplace_back(toy_throw);
    }

    for(int s = 0; s < _AnaSample_list_.size(); ++s)
    {
        const unsigned int N  = _AnaSample_list_[s]->GetN();
        const std::string det = _AnaSample_list_[s]->GetDetector();
#pragma omp parallel for num_threads(_nb_threads_)
        for(unsigned int i = 0; i < N; ++i)
        {
            AnaEvent* ev = _AnaSample_list_[s]->GetEvent(i);
            ev->ResetEvWght();
            for(int j = 0; j < _AnaFitParameters_list_.size(); ++j)
                _AnaFitParameters_list_[j]->ReWeight(ev, det, s, i, fitpar_throw[j]);
        }

        _AnaSample_list_[s]->FillEventHist(kAsimov, stat_fluc);
        _AnaSample_list_[s]->FillEventHist(kReset);
        chi2_stat += _AnaSample_list_[s]->CalcChi2();
    }

    LogInfo << "Generated toy throw from parameters.\n"
              << "Initial Chi2 Syst: " << chi2_syst << std::endl
              << "Initial Chi2 Stat: " << chi2_stat << std::endl;

    SaveParams(fitpar_throw);
}

// Loops over all samples and all bins therein, then resets event weights based on current fit parameter values, updates m_hpred, m_hmc, m_hmc_true and m_hsig histograms accordingly and computes the chi2 value which is returned:
double ND280Fitter::FillSamples(std::vector<std::vector<double>>& new_pars, int datatype)
{
    // Initialize chi2 variable which will be updated below and then returned:
    double chi2      = 0.0;

    // If the output_chi2 flag is true the chi2 contributions from the different samples are printed:
    bool output_chi2 = false;

    // Print chi2 contributions for the first 19 function calls then for every 100th function call for the first 1000 function calls and then every 1000th function call:
    if((_nb_fit_calls_ < 1001 && (_nb_fit_calls_ % 100 == 0 || _nb_fit_calls_ < 20))
       || (_nb_fit_calls_ > 1001 && _nb_fit_calls_ % 1000 == 0))
        output_chi2 = true;

    // par_offset stores the number of fit parameters for all parameter types:
    unsigned int par_offset = 0;

    // Loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(int i = 0; i < _AnaFitParameters_list_.size(); ++i)
    {
        // If we performed an eigendecomposition for this parameter type, we change the eigendecomposed input parameters back to the original parameters:
        if(_AnaFitParameters_list_[i]->IsDecomposed())
        {
            new_pars[i] = _AnaFitParameters_list_[i]->GetOriginalParameters(new_pars[i]);
        }

        // Update number of fit parameters as we loop through the different parameter types:
        par_offset += _AnaFitParameters_list_[i]->GetNpar();
    }

    // Loop over the different selection samples defined in the .json config file:
    for(int iSample = 0; iSample < _AnaSample_list_.size(); ++iSample)
    {
        // Get number of events within the current sample:
        const unsigned int num_events = _AnaSample_list_[iSample]->GetN();

        // Get the name of the detector for the current sample (as defined in the .json config file):
        const std::string detectorName = _AnaSample_list_[iSample]->GetDetector();

        // Loop over all events in the current sample (this loop will be divided amongst the different __nb_threads__):
        #pragma omp parallel for num_threads(_nb_threads_)
        for(unsigned int iEvent = 0; iEvent < num_events; ++iEvent)
        {
            // Get ith event (which contains the event information such as topology, reaction, truth/reco variables, event weights, etc.):
            AnaEvent* anaEvent = _AnaSample_list_[iSample]->GetEvent(iEvent);

            // Reset the event weight to the original one from Highland:
            anaEvent->ResetEvWght();

            // Loop over all the different parameter types such as [template, flux, detector, cross section]:
            for(int jParam = 0; jParam < _AnaFitParameters_list_.size(); ++jParam)
            {
                // Multiply the current event weight for event anaEvent with the paramter of the current parameter type for the (truth bin)/(reco bin)/(energy bin) that this event falls in:
                _AnaFitParameters_list_[jParam]->ReWeight(anaEvent, detectorName, iSample, iEvent, new_pars[jParam]);
            }
        }

        // Reset m_hpred, m_hmc, m_hmc_true and m_hsig and then fill them with the updated events:
        _AnaSample_list_[iSample]->FillEventHist(datatype);

        // Compute chi2 for the current sample (done with AnaSample::CalcLLH):
        //double sample_chi2 = _AnaSample_list_[iSample]->CalcChi2();
        double sample_chi2 = _AnaSample_list_[iSample]->CalcLLH();
        //double sample_chi2 = _AnaSample_list_[iSample]->CalcEffLLH();

        // Add the chi2 contribution from the current sample to the total chi2 variable:
        chi2 += sample_chi2;

        // If output_chi2 has been set to true before, the chi2 contribution from this sample is printed:
        if(output_chi2)
        {
            LogInfo << "Chi2 for sample " << _AnaSample_list_[iSample]->GetName() << " is "
                      << sample_chi2 << std::endl;
        }
    }

    // The total chi2 value is returned:
    return chi2;
}

// Function which is called in each iteration of the fitter to calculate and return chi2_stat + chi2_sys + chi2_reg:
double ND280Fitter::CalcLikelihood(const double* par)
{
    GenericToolbox::getElapsedTimeSinceLastCallStr(11);

    // Increase the number of function calls by 1:
    _nb_fit_calls_++;

    // If the output_chi2 flag is true the chi2 contributions from the different parameter types such as [template, flux, detector, cross section] are printed:
    bool output_chi2 = false;

    // Print chi2 contributions for the first 19 function calls then for every 100th function call for the first 1000 function calls and then every 1000th function call:
    if((_nb_fit_calls_ < 1001 && (_nb_fit_calls_ % 100 == 0 || _nb_fit_calls_ < 20))
       || (_nb_fit_calls_ > 1001 && _nb_fit_calls_ % 1000 == 0))
        output_chi2 = true;

    // Index for looping over fit parameters of this parameter type:
    int k           = 0;

    // Penalty terms for systematic uncertainties and regularization:
    double chi2_sys = 0.0;
    double chi2_reg = 0.0;

    // Vector to store all the parameter values of all parameter types:
    std::vector<std::vector<double>> new_pars;

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(int i = 0; i < _AnaFitParameters_list_.size(); ++i)
    {
        // Number of fit parameters for this parameter type:
        const unsigned int npar = _AnaFitParameters_list_[i]->GetNpar();

        // vec stores the parameter values of this iteration for this parameter type:
        std::vector<double> vec;

        // Loop over fit parameters for this parameter type:
        for(int j = 0; j < npar; ++j)
        {
            // Fill vec with the parameter values of this iteration for this parameter type:
            vec.emplace_back(par[k++]);
        }

        // If we are not using zero systematics, the systematic chi2 value is computed with AnaFitParameters::GetChi2 for this parameter type and added to chi2_sys:
        if(!_disable_syst_fit_)
            chi2_sys += _AnaFitParameters_list_[i]->GetChi2(vec);

        // If we are using regularization, the regularization chi2 value is computed with FitParameters::CalcRegularisation for this parameter type and added to chi2_reg:
        if(_AnaFitParameters_list_[i]->IsRegularised())
            chi2_reg += _AnaFitParameters_list_[i]->CalcRegularisation(vec);

        // Fill new_pars with the parameter values of all parameter types:
        new_pars.emplace_back(vec);

        // If output_chi2 has been set to true before, the chi2 contribution from this parameter type is printed:
        if(output_chi2)
        {
            LogInfo << "Chi2 contribution from " << _AnaFitParameters_list_[i]->GetName() << " is "
                      << _AnaFitParameters_list_[i]->GetChi2(vec) << std::endl;
        }
    }

    // Reset event weights based on current fit parameter values, update m_hpred, m_hmc, m_hmc_true and m_hsig histograms accordingly and compute the chi2_stat value:
    double chi2_stat = FillSamples(new_pars, kMC);

    // The different chi2 values for the current iteration of the fitter are stored in the corresponding vectors:
    vec_chi2_stat.emplace_back(chi2_stat);
    vec_chi2_sys.emplace_back(chi2_sys);
    vec_chi2_reg.emplace_back(chi2_reg);

    // If the m_save flag has been set to true, the fit parameters for the current iteration are saved with the given frequency m_freq:
    if(_nb_fit_calls_ % _save_fit_params_frequency_ == 0 && _save_fit_params_)
    {
        SaveParams(new_pars);
        SaveEventHist(_nb_fit_calls_);
    }

    // If output_chi2 has been set to true before, the different chi2 contributions are printed:
    if(output_chi2)
    {
        LogInfo << "Func Calls: " << _nb_fit_calls_ << std::endl;
        LogInfo << "Chi2 total: " << chi2_stat + chi2_sys + chi2_reg << std::endl;
        LogInfo << "Chi2 stat : " << chi2_stat << std::endl
                  << "Chi2 syst : " << chi2_sys  << std::endl
                  << "Chi2 reg  : " << chi2_reg  << std::endl;
    }

    LogDebug << "Call " << _nb_fit_calls_ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(11) << std::endl;

    // The total chi2 value is returned:
    return chi2_stat + chi2_sys + chi2_reg;
}

void ND280Fitter::SaveEventHist(int fititer, bool is_final)
{
    for(int s = 0; s < _AnaSample_list_.size(); s++)
    {
        std::stringstream ss;
        ss << "evhist_sam" << s;
        if(is_final)
            ss << "_finaliter";
        else
            ss << "_iter" << _nb_fit_calls_;

        _AnaSample_list_[s]->Write(_output_tdirectory_, ss.str(), fititer);
    }
}

void ND280Fitter::SaveEventTree(std::vector<std::vector<double>>& res_params)
{
    outtree = new TTree("selectedEvents", "selectedEvents");
    InitOutputTree();

    for(size_t s = 0; s < _AnaSample_list_.size(); s++)
    {
        for(int i = 0; i < _AnaSample_list_[s]->GetN(); i++)
        {
            AnaEvent* ev = _AnaSample_list_[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            for(size_t j = 0; j < _AnaFitParameters_list_.size(); j++)
            {
                const std::string det = _AnaSample_list_[s]->GetDetector();
                _AnaFitParameters_list_[j]->ReWeight(ev, det, s, i, res_params[j]);
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
    _output_tdirectory_->cd();
    outtree->Write();
}

void ND280Fitter::SaveParams(const std::vector<std::vector<double>>& new_pars)
{
    std::vector<double> temp_vec;
    for(size_t i = 0; i < _AnaFitParameters_list_.size(); i++)
    {
        const unsigned int npar = _AnaFitParameters_list_[i]->GetNpar();
        const std::string name  = _AnaFitParameters_list_[i]->GetName();
        std::stringstream ss;

        ss << "hist_" << name << "_iter" << _nb_fit_calls_;
        TH1D h_par(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        std::vector<std::string> vec_names;
        _AnaFitParameters_list_[i]->GetParNames(vec_names);
        for(int j = 0; j < npar; j++)
        {
            h_par.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par.SetBinContent(j + 1, new_pars[i][j]);
            temp_vec.emplace_back(new_pars[i][j]);
        }
        _output_tdirectory_->cd();
        h_par.Write();
    }

    TVectorD root_vec(temp_vec.size(), &temp_vec[0]);
    root_vec.Write(Form("vec_par_all_iter%d", _nb_fit_calls_));
}

void ND280Fitter::SaveResults(const std::vector<std::vector<double>>& par_results,
                             const std::vector<std::vector<double>>& par_errors)
{
    for(std::size_t i = 0; i < _AnaFitParameters_list_.size(); i++)
    {
        const unsigned int npar = _AnaFitParameters_list_[i]->GetNpar();
        const std::string name  = _AnaFitParameters_list_[i]->GetName();
        std::vector<double> par_original;
        _AnaFitParameters_list_[i]->GetParOriginal(par_original);

        TMatrixDSym* cov_mat = _AnaFitParameters_list_[i]->GetOriginalCovMat();

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
        _AnaFitParameters_list_[i]->GetParNames(vec_names);
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

        _output_tdirectory_->cd();
        h_par_final.Write();
        h_par_prior.Write();
        h_err_final.Write();
        h_err_prior.Write();
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
        _output_tdirectory_->cd();

        std::stringstream ss;
        ss << "par_scan_" << std::to_string(p);
        scan_graph.Write(ss.str().c_str());
    }

    delete[] x;
    delete[] y;
}

TMatrixD* ND280Fitter::GeneratePriorCovarianceMatrix(){

  std::vector<TMatrixD*> matrix_category_list;
  int nb_dof = 0;
  for(int i_parameter = 0 ; i_parameter < _AnaFitParameters_list_.size() ; i_parameter++){
    nb_dof += _AnaFitParameters_list_[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  TMatrixD* covMatrix = new TMatrixD(nb_dof,nb_dof);

  int index_shift = 0;
  for(int i_parameter = 0 ; i_parameter < _AnaFitParameters_list_.size() ; i_parameter++){
    for(int i_entry = 0 ; i_entry < _AnaFitParameters_list_[i_parameter]->GetOriginalCovMat()->GetNrows() ; i_entry++){
      for(int j_entry = 0 ; j_entry < _AnaFitParameters_list_[i_parameter]->GetOriginalCovMat()->GetNrows() ; j_entry++){
        (*covMatrix)[i_entry+index_shift][j_entry+index_shift] = (*_AnaFitParameters_list_[i_parameter]->GetOriginalCovMat())[i_entry][j_entry] ;
      }
    }
    index_shift += _AnaFitParameters_list_[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  return covMatrix;

}

TMatrixD* ND280Fitter::GeneratePosteriorCovarianceMatrix(){

  auto* covMatrix = new TMatrixD(_minimizer_->NDim(), _minimizer_->NDim() );
  _minimizer_->GetCovMatrix( covMatrix->GetMatrixArray() );
  return covMatrix;

}

