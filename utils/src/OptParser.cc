#include "OptParser.hh"

OptParser::OptParser()
{
    xsLLh_env = std::string(std::getenv("XSLLHFITTER"));

    if(xsLLh_env.empty())
    {
        std::cerr << TAG << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << TAG << "Cannot determine source tree location." << std::endl;
    }
}

bool OptParser::ParseJSON(std::string json_file)
{
    std::fstream f;
    f.open(json_file, std::ios::in);

    std::cout << TAG << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        std::cout << ERR << "Unable to open JSON configure file.\n";
        return false;
    }

    json j;
    f >> j;

    input_dir = xsLLh_env + j["input_dir"].get<std::string>();
    fname_data = input_dir + j["data_file"].get<std::string>();
    fname_mc = input_dir + j["mc_file"].get<std::string>();
    fname_output = j["output_file"].get<std::string>();

    fit_type = j["fit_type"];
    stat_fluc = j.value("stat_fluc", false);
    zero_syst = j.value("zero_syst", false);
    data_POT = j.value("data_POT", 1.0);
    mc_POT = j.value("mc_POT", 1.0);
    rng_template = j.value("rng_template", false);
    rng_seed = j.value("rng_seed", 0);
    num_threads = j.value("num_threads", 1);

    par_scan_steps = j.value("par_scan_steps", 20);
    par_scan_list = j.value("par_scan_list", std::vector<int>{});

    sample_topology = j["sample_topology"].get<std::vector<std::string>>();

    flux_cov.fname = input_dir + j["flux_cov"]["file"].get<std::string>();
    flux_cov.matrix = j["flux_cov"]["matrix"];
    flux_cov.binning = j["flux_cov"]["binning"];
    flux_cov.do_throw = j["flux_cov"]["throw"];
    flux_cov.decompose = j["flux_cov"]["decomp"];
    flux_cov.info_frac = j["flux_cov"]["variance"];
    flux_cov.do_fit = j["flux_cov"]["fit_par"];
    flux_cov.rng_start = j["flux_cov"].value("rng_start", false);

    det_cov.fname = input_dir + j["det_cov"]["file"].get<std::string>();
    det_cov.matrix = j["det_cov"]["matrix"];
    det_cov.do_throw = j["det_cov"]["throw"];
    det_cov.decompose = j["det_cov"]["decomp"];
    det_cov.info_frac = j["det_cov"]["variance"];
    det_cov.do_fit = j["det_cov"]["fit_par"];
    det_cov.rng_start = j["det_cov"].value("rng_start", false);

    xsec_cov.fname = input_dir + j["xsec_cov"]["file"].get<std::string>();
    xsec_cov.matrix = j["xsec_cov"]["matrix"];
    xsec_cov.do_throw = j["xsec_cov"]["throw"];
    xsec_cov.decompose = j["xsec_cov"]["decomp"];
    xsec_cov.info_frac = j["xsec_cov"]["variance"];
    xsec_cov.do_fit = j["xsec_cov"]["fit_par"];
    xsec_cov.rng_start = j["xsec_cov"].value("rng_start", false);

    regularise = j["regularisation"]["enable"];
    reg_strength = j["regularisation"]["strength"];
    reg_method = j["regularisation"]["method"];

    for(const auto& detector : j["detectors"])
    {
        DetOpt d;
        d.name = detector["name"];
        d.xsec = input_dir + detector["xsec_config"].get<std::string>();
        d.binning = input_dir + detector["binning"].get<std::string>();
        d.use_detector = detector["use_detector"];
        detectors.push_back(d);

        if(d.use_detector == true)
        {
            for(const auto& t : detector["template_par"])
            {
                SignalDef sd;
                sd.name = t["name"];
                sd.detector = d.name;
                sd.binning = input_dir + t["binning"].get<std::string>();
                sd.use_signal = t["use"];
                sd.definition = t["signal"].get<std::map<std::string, std::vector<int>>>();

                if(sd.use_signal)
                    signal_definition.push_back(sd);
            }
        }

    }

    for(const auto& sample : j["samples"])
    {
        SampleOpt s;
        s.cut_branch = sample["cut_branch"];
        s.name = sample["name"];
        s.detector = sample["detector"];
        s.binning = input_dir + sample["binning"].get<std::string>();
        s.use_sample = sample["use_sample"];
        samples.push_back(s);
    }

    json m;
    try
    {
        m = j.at("min_settings");
        min_settings.minimizer = m.value("minimizer", "Minuit2");
        min_settings.algorithm = m.value("algorithm", "Migrad");
        min_settings.print_level = m.value("print_level", 2);
        min_settings.strategy  = m.value("strategy", 1);
        min_settings.tolerance = m.value("tolerance", 1E-4);
        min_settings.max_iter  = m.value("max_iter", 1E6);
        min_settings.max_fcn   = m.value("max_fcn", 1E9);
    }
    catch(json::exception& e)
    {
        std::cout << TAG << "Using default minimizer settings."
                  << std::endl;

        min_settings.minimizer = "Minuit2";
        min_settings.algorithm = "Migrad";
        min_settings.print_level = 2;
        min_settings.strategy  = 1;
        min_settings.tolerance = 1E-2;
        min_settings.max_iter  = 1E6;
        min_settings.max_fcn   = 1E9;
    }

    std::cout << TAG << "Finished loading JSON configure file.\n";
    return true;
}

bool OptParser::ParseCLI(int argc, char** argv)
{
    std::cout << TAG << "Not supported yet." << std::endl;
    return false;
}

int OptParser::StringToEnum(const std::string& s) const
{
    int enum_val = -999;
    if(s == "kAsimovFit")
        enum_val = 0; //kAsimovFit;
    else if(s == "kExternalFit")
        enum_val = 1; //kExternalFit;
    else if(s == "kDataFit")
        enum_val = 2; //kDataFit;
    else if(s == "kToyFit")
        enum_val = 3; //kToyFit;
    else
        std::cout << "[WARNING] In OptParser::StringToEnum(), Invalid string!" << std::endl;

    return enum_val;
}

void OptParser::PrintOptions(bool short_list) const
{
    std::cout << TAG << "Printing parsed options..."
    << std::endl << TAG << "Data   File : " << fname_data
    << std::endl << TAG << "MC     File : " << fname_mc
    << std::endl << TAG << "Output File : " << fname_output
    << std::endl << TAG << "Fit Type : " << fit_type
    << std::endl << TAG << "Data POT : " << data_POT
    << std::endl << TAG << "MC   POT : " << mc_POT
    << std::endl << TAG << "RNG Seed : " << rng_seed
    << std::endl << TAG << "N Threads: " << num_threads
    << std::endl << TAG << "Enable Stat flucutations : " << std::boolalpha << stat_fluc
    << std::endl << TAG << "Enable Zero syst penalty : " << std::boolalpha << zero_syst
    << std::endl << TAG << "Enable Fit regularisation: " << std::boolalpha << regularise
    << std::endl;

    if(!short_list)
    {
        std::cout << TAG << "Printing more options..."
        << std::endl << TAG << "Flux Covariance file: " << flux_cov.fname
        << std::endl << TAG << "Enable flux throw : " << std::boolalpha << flux_cov.do_throw
        << std::endl << TAG << "Enable flux decomp: " << std::boolalpha << flux_cov.decompose
        << std::endl << TAG << "Det  Covariance file: " << det_cov.fname
        << std::endl << TAG << "Enable det throw : " << std::boolalpha << det_cov.do_throw
        << std::endl << TAG << "Enable det decomp: " << std::boolalpha << det_cov.decompose
        << std::endl << TAG << "Xsec Covariance file: " << xsec_cov.fname
        << std::endl << TAG << "Enable xsec throw : " << std::boolalpha << xsec_cov.do_throw
        << std::endl << TAG << "Enable xsec decomp: " << std::boolalpha << xsec_cov.decompose
        << std::endl;
    }
}
