#include "XsecCalc.hh"
using json = nlohmann::json;

XsecCalc::XsecCalc(const std::string& json_config)
    : num_toys(0), rng_seed(0), num_signals(0), total_signal_bins(0),
      postfit_cov(nullptr), postfit_cor(nullptr), toy_thrower(nullptr)
{
    std::cout << TAG << "Reading error propagation options." << std::endl;
    std::fstream f;
    f.open(json_config, std::ios::in);

    json j;
    f >> j;

    std::string input_dir = std::string(std::getenv("XSLLHFITTER"))
                            + j["input_dir"].get<std::string>();

    input_file = input_dir + j["input_fit_file"].get<std::string>();
    output_file = j["output_file"].get<std::string>();

    num_toys = j["num_toys"];
    rng_seed = j["rng_seed"];

    std::string sel_json_config = input_dir + j["sel_config"].get<std::string>();
    std::string tru_json_config = input_dir + j["tru_config"].get<std::string>();

    std::cout << TAG << "Input file from fit: " << input_file << std::endl
              << TAG << "Output xsec file: " << output_file << std::endl
              << TAG << "Num. toys: " << num_toys << std::endl
              << TAG << "RNG  seed: " << rng_seed << std::endl
              << TAG << "Selected events config: " << sel_json_config << std::endl
              << TAG << "True events config: " << tru_json_config << std::endl;

    std::cout << TAG << "Reading post-fit file..." << std::endl;
    ReadFitFile(input_file);

    std::cout << TAG << "Initializing fit objects..." << std::endl;
    selected_events = new FitObj(sel_json_config, "selectedEvents", false);
    true_events = new FitObj(tru_json_config, "trueEvents", true);
    total_signal_bins = selected_events -> GetNumSignalBins();

    InitNormalization(j["sig_norm"]);
    std::cout << TAG << "Finished initialization." << std::endl;
}

XsecCalc::~XsecCalc()
{
    delete toy_thrower;
    delete selected_events;
    delete true_events;

    delete postfit_cov;
    delete postfit_cor;
}

void XsecCalc::ReadFitFile(const std::string& file)
{
    if(postfit_cov != nullptr)
        delete postfit_cov;
    if(postfit_cor != nullptr)
        delete postfit_cor;
    postfit_param.clear();

    std::cout << TAG << "Opening " << file << std::endl;

    TFile* postfit_file = TFile::Open(file.c_str(), "READ");
    postfit_cov = (TMatrixDSym*)postfit_file -> Get("res_cov_matrix");
    postfit_cor = (TMatrixDSym*)postfit_file -> Get("res_cor_matrix");

    TVectorD* postfit_param_root = (TVectorD*)postfit_file -> Get("res_vector");
    for(int i = 0; i < postfit_param_root->GetNoElements(); ++i)
        postfit_param.emplace_back((*postfit_param_root)[i]);

    postfit_file->Close();

    std::cout << TAG << "Successfully read fit file." << std::endl;
    InitToyThrower();
}

void XsecCalc::InitToyThrower()
{
    std::cout << TAG << "Initializing toy-thrower..." << std::endl;
    if(toy_thrower != nullptr)
        delete toy_thrower;

    toy_thrower = new ToyThrower(*postfit_cov, rng_seed, 1E-48);
    if(!toy_thrower -> ForcePosDef(1E-5, 1E-48))
    {
        std::cout << ERR << "Covariance matrix could not be made positive definite.\n"
                  << "Exiting." << std::endl;
        exit(1);
    }
}

void XsecCalc::InitNormalization(const nlohmann::json& j)
{

    for(const auto& sig_def : selected_events -> GetSignalDef())
    {
        if(sig_def.use_signal == true)
        {
            std::cout << TAG << "Adding normalization parameters for "
                      << sig_def.name << " signal." << std::endl;

            json s;
            try
            {
                s = j.at(sig_def.name);
            }
            catch(json::exception& e)
            {
                std::cout << ERR << "Signal " << sig_def.name
                          << " not found in error propagation config file." << std::endl;
                exit(1);
            }

            SigNorm n;
            n.name = sig_def.name;
            n.detector = sig_def.detector;
            n.flux_file = s["flux_file"];
            n.flux_hist = s["flux_hist"];
            n.flux_int = s["flux_int"];
            n.flux_err = s["flux_err"];
            n.use_flux_fit = s["use_flux_fit"];
            n.num_targets_val = s["num_targets_val"];
            n.num_targets_err = s["num_targets_err"];
            v_normalization.push_back(n);

            std::cout << TAG << "Flux file: " << n.flux_file << std::endl
                      << TAG << "Flux hist: " << n.flux_hist << std::endl
                      << TAG << "Flux integral: " << n.flux_int << std::endl
                      << TAG << "Flux error: " << n.flux_err << std::endl
                      << TAG << "Use flux fit: " << std::boolalpha << n.use_flux_fit << std::endl
                      << TAG << "Num. targets: " << n.num_targets_val << std::endl
                      << TAG << "Num. targets err: " << n.num_targets_err << std::endl;
        }
    }
    num_signals = v_normalization.size();
}

void XsecCalc::ReweightNominal()
{
    selected_events -> ReweightNominal();
    true_events -> ReweightNominal();
}

void XsecCalc::ReweightBestFit()
{
    ReweightParam(postfit_param);

    auto sel_hists = selected_events -> GetSignalHist();
    auto tru_hists = true_events -> GetSignalHist();

    ApplyEff(sel_hists, tru_hists, false);
    ApplyNorm(sel_hists, false);

    sel_best_fit = ConcatHist(sel_hists, "sel_best_fit");
    tru_best_fit = ConcatHist(tru_hists, "tru_best_fit");
}

void XsecCalc::ReweightParam(const std::vector<double>& param)
{
    selected_events -> ReweightEvents(param);
    true_events -> ReweightEvents(param);
}

void XsecCalc::GenerateToys()
{
    GenerateToys(num_toys);
}

void XsecCalc::GenerateToys(const int ntoys)
{
    num_toys = ntoys;
    std::cout << TAG << "Throwing " << num_toys << " toys..." << std::endl;

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Throwing "));

    for(int i = 0; i < ntoys; ++i)
    {
        if(i % 10 == 0 || i == (ntoys - 1))
            pbar.Print(i, ntoys - 1);

        const unsigned int npar = postfit_param.size();
        std::vector<double> toy(npar, 0.0);
        toy_thrower -> Throw(toy);

        std::transform(toy.begin(), toy.end(), postfit_param.begin(), toy.begin(), std::plus<double>());
        for(int i = 0; i < npar; ++i)
        {
            if(toy[i] < 0.0)
                toy[i] = 0.1;
        }

        selected_events -> ReweightEvents(toy);
        true_events -> ReweightEvents(toy);

        auto sel_hists = selected_events -> GetSignalHist();
        auto tru_hists = true_events -> GetSignalHist();

        ApplyEff(sel_hists, tru_hists, true);
        ApplyNorm(sel_hists, true);

        toys_sel_events.emplace_back(ConcatHist(sel_hists, ("sel_signal_toy" + std::to_string(i))));
        toys_tru_events.emplace_back(ConcatHist(tru_hists, ("tru_signal_toy" + std::to_string(i))));
    }
}

void XsecCalc::ApplyEff(std::vector<TH1D>& sel_hist, std::vector<TH1D>& tru_hist, bool is_toy)
{
    std::vector<TH1D> eff_hist;
    for(int i = 0; i < num_signals; ++i)
    {
        TH1D eff = sel_hist[i];
        eff.Divide(&sel_hist[i], &tru_hist[i]);
        eff_hist.emplace_back(eff);

        for(int j = 1; j <= sel_hist[i].GetNbinsX(); ++j)
        {
            double bin_eff = eff.GetBinContent(j);
            double bin_val = sel_hist[i].GetBinContent(j);
            sel_hist[i].SetBinContent(j, bin_val / bin_eff);
        }
    }

    if(is_toy)
    {
        std::string eff_name = "eff_combined_toy" + std::to_string(toys_eff.size());
        toys_eff.emplace_back(ConcatHist(eff_hist, eff_name));
    }
    else
    {
        eff_best_fit = ConcatHist(eff_hist, "eff_best_fit");
    }
}

void XsecCalc::ApplyNorm(std::vector<TH1D>& vec_hist, bool is_toy)
{
    for(unsigned int i = 0; i < num_signals; ++i)
    {
        ApplyTargets(i, vec_hist[i], is_toy);
        ApplyFlux(i, vec_hist[i], is_toy);
        ApplyBinWidth(i, vec_hist[i], 1000);
    }
}

void XsecCalc::ApplyTargets(const unsigned int signal_id, TH1D& hist, bool is_toy)
{
    double num_targets = 1.0;
    if(is_toy)
    {
        num_targets = toy_thrower -> ThrowSinglePar(v_normalization[signal_id].num_targets_val,
                                                    v_normalization[signal_id].num_targets_err);
    }
    else
        num_targets = v_normalization[signal_id].num_targets_val;

    hist.Scale(1.0 / num_targets);
}

void XsecCalc::ApplyFlux(const unsigned int signal_id, TH1D& hist, bool is_toy)
{
    if(v_normalization[signal_id].use_flux_fit)
    {
    }
    else
    {
        double flux_int = 1.0;
        if(is_toy)
        {
            flux_int = toy_thrower -> ThrowSinglePar(v_normalization[signal_id].flux_int,
                                                     v_normalization[signal_id].flux_err);
        }
        else
            flux_int = v_normalization[signal_id].flux_int;

        hist.Scale(1.0 / flux_int);
    }
}

void XsecCalc::ApplyBinWidth(const unsigned int signal_id, TH1D& hist, const double unit_scale)
{
    BinManager bm = selected_events -> GetBinManager(signal_id);
    for(int i = 0; i < hist.GetNbinsX(); ++i)
    {
        const double bin_width = bm.GetBinWidth(i) / unit_scale;
        const double bin_value = hist.GetBinContent(i+1);
        hist.SetBinContent(i+1, bin_value / bin_width);
    }
}

TH1D XsecCalc::ConcatHist(const std::vector<TH1D>& vec_hist, const std::string& hist_name)
{
    TH1D hist_combined(hist_name.c_str(), hist_name.c_str(),
                       total_signal_bins, 0, total_signal_bins);

    unsigned int bin = 1;
    for(const auto& hist : vec_hist)
    {
        for(int i = 1; i <= hist.GetNbinsX(); ++i)
            hist_combined.SetBinContent(bin++, hist.GetBinContent(i));
    }

    return hist_combined;
}

void XsecCalc::CalcCovariance(bool use_best_fit)
{
    std::cout << TAG << "Calculating covariance matrix..." << std::endl;

    TH1D h_cov("h_cov", "h_cov", total_signal_bins, 0, total_signal_bins);
    if(use_best_fit)
    {
        ReweightBestFit();
        h_cov = sel_best_fit;
        std::cout << TAG << "Using best fit for covariance." << std::endl;
    }
    else
    {
        TH1D h_mean("","",total_signal_bins, 0, total_signal_bins);
        for(const auto& hist : toys_sel_events)
        {
            for(int i = 0; i < total_signal_bins; ++i)
                h_mean.Fill(i + 0.5, hist.GetBinContent(i+1));
        }
        h_mean.Scale(1.0 / (1.0 * num_toys));
        h_cov = h_mean;

        std::cout << TAG << "Using mean of toys for covariance." << std::endl;
    }

    xsec_cov.ResizeTo(total_signal_bins, total_signal_bins);
    xsec_cov.Zero();

    xsec_cor.ResizeTo(total_signal_bins, total_signal_bins);
    xsec_cor.Zero();

    for(const auto& hist : toys_sel_events)
    {
        for(int i = 0; i < total_signal_bins; ++i)
        {
            for(int j = 0; j < total_signal_bins; ++j)
            {
                const double x = hist.GetBinContent(i+1) - h_cov.GetBinContent(i+1);
                const double y = hist.GetBinContent(j+1) - h_cov.GetBinContent(j+1);
                xsec_cov(i,j) += x * y / (1.0 * num_toys);
            }
        }
    }

    for(int i = 0; i < total_signal_bins; ++i)
    {
        for(int j = 0; j < total_signal_bins; ++j)
        {
            const double x = xsec_cov(i,i);
            const double y = xsec_cov(j,j);
            const double z = xsec_cov(i,j);
            xsec_cor(i,j) = z / (sqrt(x * y));

            if(std::isnan(xsec_cor(i,j)))
                xsec_cor(i,j) = 0.0;
        }
    }

    for(int i = 0; i < total_signal_bins; ++i)
        sel_best_fit.SetBinError(i+1, sqrt(xsec_cov(i,i)));

    std::cout << TAG << "Covariance and correlation matrix calculated." << std::endl;
}

void XsecCalc::SaveOutput(const std::string& override_file, bool save_toys)
{
    TFile* file = nullptr;
    if(!override_file.empty())
    {
        file = TFile::Open(override_file.c_str(), "RECREATE");
        std::cout << TAG << "Saving output to " << override_file << std::endl;
    }
    else
    {
        file = TFile::Open(output_file.c_str(), "RECREATE");
        std::cout << TAG << "Saving output to " << output_file << std::endl;
    }


    file->cd();
    if(save_toys)
    {
        for(int i = 0; i < num_toys; ++i)
        {
            toys_sel_events.at(i).Write();
            toys_tru_events.at(i).Write();
            toys_eff.at(i).Write();
        }
    }


    sel_best_fit.Write("sel_best_fit");
    tru_best_fit.Write("tru_best_fit");
    eff_best_fit.Write("eff_best_fit");

    xsec_cov.Write("xsec_cov");
    xsec_cor.Write("xsec_cor");

    postfit_cov->Write("postfit_cov");
    postfit_cor->Write("postfit_cor");

    TVectorD postfit_param_root(postfit_param.size(), postfit_param.data());
    postfit_param_root.Write("postfit_param");

    file->Close();
}
