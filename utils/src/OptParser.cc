#include "OptParser.hh"
#include "Logger.h"
#include "GenericToolbox.h"

OptParser::OptParser()
{
    Logger::setUserHeaderStr("[OptParser]");
    xsLLh_env = std::string(std::getenv("XSLLHFITTER"));

    if(xsLLh_env.empty())
    {
        LogError << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                 << "Cannot determine source tree location." << std::endl;
    }
}

bool OptParser::ParseJSON(std::string json_file)
{
    std::fstream f;
    f.open(json_file, std::ios::in);

    LogInfo << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        LogError << "Unable to open JSON configure file.\n";
        return false;
    }

    json j;
    f >> j;

    input_dir = xsLLh_env + j["input_dir"].get<std::string>();
    fname_data = j["data_file"].get<std::string>();
    if(not fname_data.empty()) fname_data = input_dir + fname_data;
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
    save_events = j.value("save_events", true);

    par_scan_steps = j.value("par_scan_steps", 20);
    par_scan_list = j.value("par_scan_list", std::vector<int>{});

    sample_topology = j["sample_topology"].get<std::vector<std::string>>();

    // Get the Highland codes for the different topologies and store them in the topology_HL_code vector:
    topology_HL_code = j["topology_HL_code"].get<std::vector<int>>();

    flux_cov.fname = input_dir + j["flux_cov"]["file"].get<std::string>();
    flux_cov.matrix = j["flux_cov"]["matrix"];
    flux_cov.binning = input_dir + j["flux_cov"]["binning"].get<std::string>();
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


    if( j.find("sample_definitions_file") != j.end() )
    {
        LogDebug << "Reading sample definition from file: " << j["sample_definitions_file"] << std::endl;
        if( not GenericToolbox::doesPathIsFile(j["sample_definitions_file"]) ){
            LogError << "Could not find sample definition file." << std::endl;
            throw std::runtime_error("Could not find file.");
        }

        std::fstream sampleFile;
        sampleFile.open(j["sample_definitions_file"], std::ios::in);

        json sampleJson;
        sampleFile >> sampleJson;
        j["samples"] = sampleJson["samples"];
        sampleFile.close();
    }

    if( j["samples"].empty() ) {
        LogError << "No samples have been defined." << std::endl;
        throw std::logic_error("No samples have been defined.");
    }

    LogDebug << "Reading " << j["samples"].size() << " sample definitions..." << std::endl;
    for(const auto& sample : j["samples"])
    {
        SampleOpt s;
        s.cut_branch = sample["cut_branch"];
        s.name = sample["name"];
        s.detector = sample["detector"];
        s.binning = input_dir + sample["binning"].get<std::string>();
        s.use_sample = sample["use_sample"];
        if(sample.find("additional_cuts") != sample.end()) s.additional_cuts = sample["additional_cuts"].get<std::string>();
        if(sample.find("data_POT") != sample.end()) s.data_POT = sample["data_POT"].get<double>();
        if(sample.find("mc_POT") != sample.end()) s.mc_POT = sample["mc_POT"].get<double>();
        s.fit_phase_space = {"D2Reco", "D1Reco"};
        if(sample.find("fit_phase_space") != sample.end()) s.fit_phase_space = sample["fit_phase_space"].get<std::vector<std::string>>();
        samples.push_back(s);
    }

    json m;
    try
    {
        m = j.at("min_settings");
        min_settings.minimizer = m.value("minimizer", "Minuit2");
        min_settings.algorithm = m.value("algorithm", "Migrad");
        min_settings.likelihood = m.value("likelihood", "Poisson");
        min_settings.print_level = m.value("print_level", 2);
        min_settings.strategy  = m.value("strategy", 1);
        min_settings.tolerance = m.value("tolerance", 1E-4);
        min_settings.max_iter  = m.value("max_iter", 1E6);
        min_settings.max_fcn   = m.value("max_fcn", 1E9);
    }
    catch(json::exception& e)
    {
        LogInfo << "Using default minimizer settings."
                  << std::endl;

        min_settings.minimizer = "Minuit2";
        min_settings.algorithm = "Migrad";
        min_settings.likelihood = "Poisson";
        min_settings.print_level = 2;
        min_settings.strategy  = 1;
        min_settings.tolerance = 1E-2;
        min_settings.max_iter  = 1E6;
        min_settings.max_fcn   = 1E9;
    }

    LogInfo << "Finished loading JSON configure file.\n";
    return true;
}

bool OptParser::ParseCLI(int argc, char** argv)
{
    LogInfo << "Not supported yet." << std::endl;
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
        LogWarning << "In OptParser::StringToEnum(), Invalid string!" << std::endl;

    return enum_val;
}

void OptParser::PrintOptions(bool short_list) const
{
    LogInfo << "Printing parsed options..."
    << std::endl << "Data   File : " << fname_data
    << std::endl << "MC     File : " << fname_mc
    << std::endl << "Output File : " << fname_output
    << std::endl << "Fit Type : " << fit_type
    << std::endl << "Data POT : " << data_POT
    << std::endl << "MC   POT : " << mc_POT
    << std::endl << "RNG Seed : " << rng_seed
    << std::endl << "N Threads: " << num_threads
    << std::endl << "Saving Events: " << std::boolalpha << save_events
    << std::endl << "Enable Stat flucutations : " << std::boolalpha << stat_fluc
    << std::endl << "Enable Zero syst penalty : " << std::boolalpha << zero_syst
    << std::endl << "Enable Fit regularisation: " << std::boolalpha << regularise
    << std::endl;

    if(!short_list)
    {
        LogInfo << "Printing more options..."
        << std::endl << "Flux Covariance file: " << flux_cov.fname
        << std::endl << "Enable flux throw : " << std::boolalpha << flux_cov.do_throw
        << std::endl << "Enable flux decomp: " << std::boolalpha << flux_cov.decompose
        << std::endl << "Det  Covariance file: " << det_cov.fname
        << std::endl << "Enable det throw : " << std::boolalpha << det_cov.do_throw
        << std::endl << "Enable det decomp: " << std::boolalpha << det_cov.decompose
        << std::endl << "Xsec Covariance file: " << xsec_cov.fname
        << std::endl << "Enable xsec throw : " << std::boolalpha << xsec_cov.do_throw
        << std::endl << "Enable xsec decomp: " << std::boolalpha << xsec_cov.decompose
        << std::endl;
    }
}
