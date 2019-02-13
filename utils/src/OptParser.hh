#ifndef OPTPARSER_HH
#define OPTPARSER_HH

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//#include "XsecFitter.hh"
#include "ColorOutput.hh"

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

struct SignalDef
{
    bool use_signal;
    std::string name;
    std::string detector;
    std::string binning;
    std::map<std::string, std::vector<int>> definition;
};

struct CovOpt
{
    std::string fname;
    std::string matrix;
    std::string binning;
    bool do_throw;
    bool decompose;
    double info_frac;
};

struct DetOpt
{
    std::string name;
    std::string xsec;
    std::string binning;
    bool use_detector;
};

struct MinSettings
{
    std::string minimizer;
    std::string algorithm;
    int print_level;
    int strategy;
    double tolerance;
    double max_iter;
    double max_fcn;
};

class OptParser
{
    public:

        OptParser();
        bool ParseJSON(std::string json_file);
        bool ParseCLI(int argc, char** argv);

        int StringToEnum(const std::string& s) const;
        void PrintOptions(bool short_list = true) const;

        std::string fname_data;
        std::string fname_mc;
        std::string fname_output;
        std::string input_dir;
        std::string xsLLh_env;

        int fit_type;
        int rng_seed;
        int num_threads;
        double data_POT;
        double mc_POT;

        bool stat_fluc;
        bool zero_syst;
        bool regularise;
        double reg_strength;
        std::string reg_method;

        std::vector<std::string> sample_topology;
        std::vector<SignalDef> signal_definition;

        CovOpt flux_cov;
        CovOpt det_cov;
        CovOpt xsec_cov;
        MinSettings min_settings;
        std::vector<SampleOpt> samples;
        std::vector<DetOpt> detectors;

    private:

        const std::string TAG = color::GREEN_STR + "[OptParser]: " + color::RESET_STR;
        const std::string ERR = color::RED_STR + color::BOLD_STR
                                + "[ERROR]: " + color::RESET_STR;
};
#endif
