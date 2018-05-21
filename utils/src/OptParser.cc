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

    data_POT = j["data_POT"];
    mc_POT = j["mc_POT"];
    rng_seed = j["rng_seed"];
    num_threads = j["num_threads"];

    sample_signal = j["sample_signal"].get<std::vector<int> >();
    sample_topology = j["sample_topology"].get<std::vector<std::string> >();

    flux_cov.fname = input_dir + j["flux_cov"]["file"].get<std::string>();
    flux_cov.matrix = j["flux_cov"]["matrix"];
    flux_cov.binning = j["flux_cov"]["binning"];

    det_cov.fname = input_dir + j["det_cov"]["file"].get<std::string>();
    det_cov.matrix = j["det_cov"]["matrix"];
    det_cov.binning = j["det_cov"]["binning"];

    for(const auto& sample : j["samples"])
    {
       SampleOpt s;
       s.cut_branch = sample["cut_branch"];
       s.name = sample["name"];
       s.detector = sample["detector"];
       s.binning = input_dir + sample["binning"].get<std::string>();
       s.flux_offset = sample["flux_offset"];
       s.det_offset = sample["det_offset"];
       s.use_sample = sample["use_sample"];
       samples.push_back(s);
    }

    std::cout << "[OptParser]: Finished loading JSON configure file.\n";
    return true;
}

bool OptParser::ParseCLI(int argc, char** argv)
{
    return false;
}
