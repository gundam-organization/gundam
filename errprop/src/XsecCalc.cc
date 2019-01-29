#include "XsecCalc.hh"
using json = nlohmann::json;

XsecCalc::XsecCalc(const std::string& json_config)
    : num_toys(0), rng_seed(0), postfit_cov(nullptr), postfit_cor(nullptr), toy_thrower(nullptr)
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

    InitNormalization(j["sig_norm"]);
    std::cout << TAG << "Finished initialization." << std::endl;
}

XsecCalc::~XsecCalc()
{
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
}

void XsecCalc::ReweightNominal()
{
    selected_events -> ReweightNominal();
    true_events -> ReweightNominal();
}

void XsecCalc::ReweightBestFit()
{
    ReweightParam(postfit_param);
    sel_best_fit = selected_events -> GetHistCombined("best_fit");
    tru_best_fit = true_events -> GetHistCombined("best_fit");

    eff_best_fit = sel_best_fit;
    eff_best_fit.Divide(&sel_best_fit, &tru_best_fit);
    eff_best_fit.SetName("eff_best_fit");
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
    for(int i = 0; i < ntoys; ++i)
    {
        const unsigned int npar = postfit_param.size();
        std::vector<double> toy(npar, 0.0);
        toy_thrower -> Throw(toy);

        std::transform(toy.begin(), toy.end(), postfit_param.begin(), toy.begin(), std::plus<double>());
        for(int i = 0; i < npar; ++i)
        {
            if(toy[i] < 0.0)
                toy[i] = 0.0;
        }

        selected_events -> ReweightEvents(toy);
        true_events -> ReweightEvents(toy);

        std::string suffix = "toy" + std::to_string(i);
        toys_sel_events.emplace_back(selected_events->GetHistCombined(suffix));
        toys_tru_events.emplace_back(true_events->GetHistCombined(suffix));

        toys_eff.emplace_back(toys_sel_events.at(i));
        toys_eff.at(i).Divide(&toys_sel_events[i], &toys_tru_events[i]);
        toys_eff.at(i).SetName(("eff_" + suffix).c_str());
    }
}

void XsecCalc::SaveOutput(const std::string& override_file)
{
    TFile* file = nullptr;
    if(!override_file.empty())
        file = TFile::Open(override_file.c_str(), "RECREATE");
    else
        file = TFile::Open(output_file.c_str(), "RECREATE");


    file->cd();
    for(int i = 0; i < num_toys; ++i)
    {
        toys_sel_events.at(i).Write();
        toys_tru_events.at(i).Write();
        toys_eff.at(i).Write();
    }

    sel_best_fit.Write("sel_best_fit");
    tru_best_fit.Write("tru_best_fit");
    eff_best_fit.Write("eff_best_fit");

    postfit_cov->Write("postfit_cov");
    postfit_cor->Write("postfit_cor");

    TVectorD postfit_param_root(postfit_param.size(), postfit_param.data());
    postfit_param_root.Write("postfit_param");

    file->Close();
}
