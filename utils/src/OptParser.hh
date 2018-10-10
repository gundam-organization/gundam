#ifndef OPTPARSER_HH
#define OPTPARSER_HH

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

struct SampleOpt
{
    int cut_branch;
    bool use_sample;
    std::string name;
    std::string detector;
    std::string binning;
};

struct CovOpt
{
    std::string fname;
    std::string matrix;
    std::string binning;
    bool decompose;
    double info_frac;
};

struct DetOpt
{
    std::string name;
    std::string xsec;
    std::string binning;
    std::string flux_file;
    std::string flux_hist;
    double flux_integral;
    double flux_error;
    double ntargets_val;
    double ntargets_err;
    bool use_detector;
};

class OptParser
{
    public:

        OptParser();
        bool ParseJSON(std::string json_file);
        bool ParseCLI(int argc, char** argv);

        std::string fname_data;
        std::string fname_mc;
        std::string fname_output;
        std::string fname_xsec;
        std::string input_dir;
        std::string xsLLh_env;

        int data_POT;
        int mc_POT;
        int rng_seed;
        int num_threads;
        int num_throws;

        bool regularise;
        double reg_strength;
        std::string reg_method;

        std::vector<int> sample_signal;
        std::vector<std::string> sample_topology;

        CovOpt flux_cov;
        CovOpt det_cov;
        CovOpt xsec_cov;
        std::vector<SampleOpt> samples;
        std::vector<DetOpt> detectors;
};
#endif
