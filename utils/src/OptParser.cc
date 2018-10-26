#include "OptParser.hh"

OptParser::OptParser()
{
    xsLLh_env = std::string(std::getenv("XSLLHFITTER"));

    if(xsLLh_env.empty())
    {
        std::cerr << "[ERROR]: Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << "[ERROR]: Cannot determine source tree location." << std::endl;
    }
}

bool OptParser::ParseJSON(std::string json_file)
{
    std::fstream f;
    f.open(json_file, std::ios::in);

    std::cout << "[OptParser]: Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        std::cout << "[ERROR] Unable to open JSON configure file.\n";
        return false;
    }

    json j;
    f >> j;

    input_dir = xsLLh_env + j["input_dir"].get<std::string>();
    fname_data = input_dir + j["data_file"].get<std::string>();
    fname_mc = input_dir + j["mc_file"].get<std::string>();
    fname_output = j["output_file"].get<std::string>();
    fname_xsec = j["xsec_file"].get<std::string>();

    fit_type = j["fit_type"];
    stat_fluc = j["stat_fluc"];
    zero_syst = j["zero_syst"];
    data_POT = j["data_POT"];
    mc_POT = j["mc_POT"];
    rng_seed = j["rng_seed"];
    num_threads = j["num_threads"];
    num_throws = j["num_throws"];

    sample_signal = j["sample_signal"].get<std::vector<int> >();
    sample_topology = j["sample_topology"].get<std::vector<std::string> >();

    flux_cov.fname = input_dir + j["flux_cov"]["file"].get<std::string>();
    flux_cov.matrix = j["flux_cov"]["matrix"];
    flux_cov.binning = j["flux_cov"]["binning"];
    flux_cov.decompose = j["flux_cov"]["decomp"];
    flux_cov.info_frac = j["flux_cov"]["variance"];

    det_cov.fname = input_dir + j["det_cov"]["file"].get<std::string>();
    det_cov.matrix = j["det_cov"]["matrix"];
    det_cov.binning = j["det_cov"]["binning"];
    det_cov.decompose = j["det_cov"]["decomp"];
    det_cov.info_frac = j["det_cov"]["variance"];

    xsec_cov.fname = input_dir + j["xsec_cov"]["file"].get<std::string>();
    xsec_cov.matrix = j["xsec_cov"]["matrix"];
    xsec_cov.decompose = j["xsec_cov"]["decomp"];
    xsec_cov.info_frac = j["xsec_cov"]["variance"];

    regularise = j["regularisation"]["enable"];
    reg_strength = j["regularisation"]["strength"];
    reg_method = j["regularisation"]["method"];

    for(const auto& detector : j["detectors"])
    {
        DetOpt d;
        d.name = detector["name"];
        d.xsec = input_dir + detector["xsec_config"].get<std::string>();
        d.binning = input_dir + detector["binning"].get<std::string>();
        d.flux_file = input_dir + detector["flux_file"].get<std::string>();
        d.flux_hist = detector["flux_hist"];
        d.flux_integral = detector["flux_integral"];
        d.flux_error = detector["flux_error"];
        d.ntargets_val = detector["ntargets_val"];
        d.ntargets_err = detector["ntargets_err"];
        d.use_detector = detector["use_detector"];
        detectors.push_back(d);
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

    std::cout << "[OptParser]: Finished loading JSON configure file.\n";
    return true;
}

bool OptParser::ParseCLI(int argc, char** argv)
{
    std::cout << "[OptParser]: Not supported yet." << std::endl;
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
