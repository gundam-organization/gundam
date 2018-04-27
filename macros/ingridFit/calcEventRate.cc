struct FitBin
{
    double D1low, D1high;
    double D2low, D2high;

    FitBin() : D1low(0), D1high(0), D2low(0), D2high(0) {}
    FitBin(const double& D1_L, const double& D1_H,
           const double& D2_L, const double& D2_H)
          : D1low(D1_L), D1high(D1_H),
            D2low(D2_L), D2high(D2_H)
          {}
};

int GetBinIndex(double D1, double D2, const std::vector<FitBin>& bins)
{
    for(int i = 0; i < bins.size(); ++i)
    {
        if(D1 >= bins[i].D1low && D1 < bins[i].D1high &&
           D2 >= bins[i].D2low && D2 < bins[i].D2high)
        {
            return i;
        }
    }
    return 0;
}

void SetBinning(const std::string& file_name, std::vector<FitBin>& bins)
{
    std::ifstream fin(file_name.c_str(), std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: In SetBinning()\n"
                  << "[ERROR]: Failed to open binning file: " << file_name << std::endl;
        return;
    }

    else
    {
        std::string line;
        while(getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
            {
                std::cout << "[WARNING]: In FitParameters::SetBinning()\n"
                          << "[WARNING]: Bad line format: " << line << std::endl;
                continue;
            }
            bins.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        fin.close();
    }
}

void calcEventRate(const std::string fname_mc, const std::string fname_data, const std::string fname_fit, const std::string fname_output, const std::string fname_bins, double pot_ratio)
{
    TFile* file_mc   = TFile::Open(fname_mc.c_str(), "READ");
    TFile* file_data = TFile::Open(fname_data.c_str(), "READ");
    TFile* file_fit  = TFile::Open(fname_fit.c_str(), "READ");
    TFile* file_out  = TFile::Open(fname_output.c_str(), "RECREATE");

    TTree* t_sel_events_mc = (TTree*)file_mc -> Get("selectedEvents");
    TTree* t_sel_events_data = (TTree*)file_data -> Get("selectedEvents");
    TTree* t_true_events_mc = (TTree*)file_mc -> Get("trueEvents");
    TTree* t_true_events_data = (TTree*)file_data -> Get("trueEvents");

    TH1D* fit_param_values = (TH1D*)file_fit -> Get("paramhist_parpar_fit_result");
    TH1D* fit_param_errors = (TH1D*)file_fit -> Get("paramerrhist_parpar_fit_result");

    const int nbins_sample = fit_param_values -> GetXaxis() -> GetNbins();
    const TArrayD* bins_sample = fit_param_values -> GetXaxis() -> GetXbins();

    TH1D* total_mc_sample = new TH1D("total_mc_sample", "total_mc_sample", nbins_sample, bins_sample -> GetArray());
    TH1D* total_data_sample = new TH1D("total_data_sample", "total_data_sample", nbins_sample, bins_sample -> GetArray());
    TH1D* total_fit_sample = new TH1D("total_fit_sample", "total_fit_sample", nbins_sample, bins_sample -> GetArray());

    TH1D* nd280_mc_sample = (TH1D*)file_fit -> Get("evhist_sam0_iter0_mc");
    TH1D* nd280_data_sample = (TH1D*)file_fit -> Get("evhist_sam0_iter0_data");
    TH1D* nd280_fit_sample = (TH1D*)file_fit -> Get("evhist_sam0_finaliter_pred");

    TH1D* ingrid_mc_sample = (TH1D*)file_fit -> Get("evhist_sam1_iter0_mc");
    TH1D* ingrid_data_sample = (TH1D*)file_fit -> Get("evhist_sam1_iter0_data");
    TH1D* ingrid_fit_sample = (TH1D*)file_fit -> Get("evhist_sam1_finaliter_pred");

    total_mc_sample -> Add(nd280_mc_sample, ingrid_mc_sample);
    total_data_sample -> Add(nd280_data_sample, ingrid_data_sample);
    total_fit_sample -> Add(nd280_fit_sample, ingrid_fit_sample);

    TH1D* signal_mc = new TH1D("signal_mc", "signal_mc", nbins_sample, 0, nbins_sample);
    TH1D* signal_data = new TH1D("signal_data", "signal_data", nbins_sample, 0, nbins_sample);
    TH1D* signal_fit = new TH1D("signal_fit", "signal_fit", nbins_sample, 0, nbins_sample);

    std::vector<FitBin> analysis_bins;
    SetBinning(fname_bins, analysis_bins);

    float D1True = 0;
    float D2True = 0;
    float weight = 0;
    int mectopology = -1;
    int cutBranch = -1;

    t_sel_events_mc -> SetBranchAddress("D1True", &D1True);
    t_sel_events_mc -> SetBranchAddress("D2True", &D2True);
    t_sel_events_mc -> SetBranchAddress("mectopology", &mectopology);
    t_sel_events_mc -> SetBranchAddress("cutBranch", &cutBranch);
    t_sel_events_mc -> SetBranchAddress("weight", &weight);

    int nentries = t_sel_events_mc -> GetEntriesFast();
    for(int i = 0; i < nentries; ++i)
    {
        t_sel_events_mc -> GetEntry(i);

        if((mectopology == 1 || mectopology == 2) && cutBranch == 1)
        {
            int idx = GetBinIndex(D1True, D2True, analysis_bins);
            signal_mc -> Fill(idx, weight);
        }
    }
    signal_mc -> Scale(pot_ratio);

    t_sel_events_data -> SetBranchAddress("D1True", &D1True);
    t_sel_events_data -> SetBranchAddress("D2True", &D2True);
    t_sel_events_data -> SetBranchAddress("mectopology", &mectopology);
    t_sel_events_data -> SetBranchAddress("cutBranch", &cutBranch);
    t_sel_events_data -> SetBranchAddress("weight", &weight);

    nentries = t_sel_events_data -> GetEntriesFast();
    for(int i = 0; i < nentries; ++i)
    {
        t_sel_events_data -> GetEntry(i);

        if((mectopology == 1 || mectopology == 2) && cutBranch == 1)
        {
            int idx = GetBinIndex(D1True, D2True, analysis_bins);
            signal_data -> Fill(idx, weight);
        }
    }

    for(int i = 1; i < nbins_sample+1; ++i)
    {
        signal_fit -> SetBinContent(i, signal_mc -> GetBinContent(i) * fit_param_values -> GetBinContent(i));
        signal_fit -> SetBinError(i, signal_fit -> GetBinContent(i) * fit_param_errors -> GetBinContent(i));
    }

    file_out -> cd();
    signal_mc -> Write();
    signal_data -> Write();
    signal_fit -> Write();

    return;
}
