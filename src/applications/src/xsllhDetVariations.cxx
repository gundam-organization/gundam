#include <TMath.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TStyle.h"
#include "TTree.h"
#include "TVectorT.h"

#include <GenericToolbox.h>
#include <Logger.h>
#include "OptParser.hh"
#include <TFormula.h>
#include <TLeaf.h>
#include <TTreeFormula.h>

#include "json.hpp"
using json = nlohmann::json;

#include "BinManager.hh"
#include "ColorOutput.hh"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"
#include "ProgressBar.hh"

// Structure that holds the options and the binning for a file specified in the .json config file:
struct FileOptions {

    FileOptions() = default;

    std::string fname_input;
    std::string tree_name;
    std::string detector;
    unsigned int num_samples{};
    unsigned int num_toys{};
    unsigned int num_syst{};
    std::vector<int> cuts;
    std::map<int, std::vector<int>> samples;

    // Vector which will hold the mapping of variable names to branch number for all variables (e.g., reco muon momentum and reco muon costheta):
    std::vector<std::map<std::string, std::vector<int>>> variable_mapping;

    // Number of variable names:
    unsigned int num_var_names{};

    // Names of all variables:
    std::vector<std::string> variable_names;

    std::vector<BinManager> bin_manager;

    FileOptions(const FileOptions& source_){

        fname_input = source_.fname_input;
        tree_name = source_.tree_name;
        detector = source_.detector;

        num_samples = source_.num_samples;
        num_toys = source_.num_toys;
        num_syst = source_.num_syst;

        for(const auto& cut: source_.cuts ) cuts.emplace_back(cut);
        for(const auto& sample: source_.samples ) samples[sample.first] = sample.second;
        for(const auto& a_variable_mapping: source_.variable_mapping ){
            variable_mapping.emplace_back(std::map<std::string, std::vector<int>>());
            for(const auto& var: a_variable_mapping){
                variable_mapping.back()[var.first] = std::vector<int>();
                for(const auto& map: var.second){
                    variable_mapping.back()[var.first].emplace_back(map);
                }
            }
        }

        num_var_names = source_.num_var_names;

        for(const auto& variable_name: source_.variable_names){
            variable_names.emplace_back(variable_name);
        }

        for(const auto& bin: source_.bin_manager){
            bin_manager.emplace_back(bin);
        }

    }

};

std::map<std::string, std::function<double(TTree* tree)>> __cutDictionnary__;
int __currentFGD__;
TFile* __currentInputFile__;
TTree* __currentInputTree__;

std::vector<SampleOpt> sampleList;
std::vector<TTreeFormula*> cutFormulaList;
TFile* __treeConverterFile__ = nullptr;
TTree* __treeConverterRecoTTree__ = nullptr;
std::map<int, std::map<int, std::map<int, Int_t>>> __treeConvEntryToVertexIDSplit__; // run, subrun, entry, vertexID
std::map<int, int> __currentFlatToTreeConvEntryMapping__;

struct AdditionalCutParser{

    AdditionalCutParser() = default;

    void initializeFormula(){

        if(cutString.empty()){
            LogFatal << "Can't initialize formula: cutString is empty." << std::endl;
            exit(EXIT_FAILURE);
        }

        // parsing cutString
        int parameterId = 0;
        std::stringstream ss;
        std::string parsedCutString = cutString;
        for(auto& cutDictPair: __cutDictionnary__){
            ss << "[" << parameterId << "]";
            std::string newParsedCutStr = GenericToolbox::replaceSubstringInString(parsedCutString, cutDictPair.first, ss.str());
            if(newParsedCutStr != parsedCutString){

            }
        }

    }

    std::string cutString;
    std::vector<std::string> parameterNameList;
    TFormula* cutFormula = nullptr;

};

bool doesEntryPassAdditionalCut(TTree* tree_, int sampleId_);
void initCutDictionnary();
void mapTreeConverterEntries();
bool buildTreeSyncCache();

int main(int argc, char** argv)
{

    Logger::setUserHeaderStr("[xsDetVariation]");
    initCutDictionnary();

    // Define colors and strings for info and error messages:
//    const std::string TAG = color::GREEN_STR + "[xsDetVariation]: " + color::RESET_STR;
//    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    // Print welcome message:
    LogInfo << "--------------------------------------------------------\n"
              << color::RainbowText("Welcome to the Super-xsLLh Detector Variation Interface.\n")
              << color::RainbowText("Initializing the variation machinery...") << std::endl;

    // Progress bar for reading in events from the ROOT file:
    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(Logger::getPrefixString() + "Reading Events "));

    // .json config file that will be parsed from the command line:
    std::string json_file;
    std::string sampleDefFile;
    int rebin_factor = 1;

    for(int iArg = 0 ; iArg < argc ; iArg++){
        if(std::string(argv[iArg]) == "-j"){
            int jArg = iArg + 1;
            if (jArg < argc) {
                json_file = std::string(argv[jArg]);
            }
            else {
                LogError << "Give an argument after " << argv[iArg] << std::endl;
                throw std::logic_error(std::string(argv[iArg]) + " : no argument found");
            }
        }
        else if(std::string(argv[iArg]) == "-s"){
            int jArg = iArg + 1;
            if (jArg < argc) {
                sampleDefFile = std::string(argv[jArg]);
            }
            else {
                LogError << "Give an argument after " << argv[iArg] << std::endl;
                throw std::logic_error(std::string(argv[iArg]) + " : no argument found");
            }
        }
        else if(std::string(argv[iArg]) == "-r"){
            int jArg = iArg + 1;
            if (jArg < argc) {
                rebin_factor = std::stoi(argv[jArg]);
            }
            else {
                LogError << "Give an argument after " << argv[iArg] << std::endl;
                throw std::logic_error(std::string(argv[iArg]) + " : no argument found");
            }
        }
        else if(std::string(argv[iArg]) == "-h"){
            std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
                      << "-j : JSON input\n";
        }
    }

    // Read in the .json config file:
    std::fstream f;
    f.open(json_file, std::ios::in);
    LogInfo << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        LogError << "Unable to open JSON configure file." << std::endl;
        return 1;
    }

    json j;
    f >> j;

    // Get the configuration from the .json config file:
    bool do_mc_stat     = j["mc_stat_error"];
    bool do_projection  = j["projection"];
    bool do_single_syst = j["single_syst"];
    bool do_covariance  = j["covariance"];
    bool do_print       = j["pdf_print"];

    unsigned int syst_idx   = j["syst_idx"];
    const double weight_cut = j["weight_cut"];

    std::string fname_output  = j["fname_output"];
    std::string cov_mat_name  = j["covariance_name"];
    std::string cor_mat_name  = j["correlation_name"];

    std::string variable_plot;
    if(j.find("plot_variable") != j.end()) variable_plot = j["plot_variable"].get<std::string>();

    std::vector<std::string> var_names = j["var_names"].get<std::vector<std::string>>();
    const int nvars = var_names.size();

    std::vector<int> sampleCuts;
    std::vector<int> sampleRebin;

    // cov_bin_manager will hold the binnings of the different samples (the length of this vector will be equal to the number of samples):
    std::vector<BinManager> cov_bin_manager;
    std::map<std::string, std::string> temp_cov_binning;

    // Number of samples:
    unsigned int num_cov_samples;

    if( sampleDefFile.empty() and j.find("sample_definition_config") != j.end()){
        sampleDefFile = j["sample_definition_config"].get<std::string>();
    }



    if( not sampleDefFile.empty() ){

        if(j.find("sample_cuts") != j.end()){
            sampleCuts = j["sample_cuts"].get<std::vector<int>>();
        }
        else{
            LogFatal << "Could not find \"sample_cuts\" parameter in the config file." << std::endl;
            LogFatal << "This parameter is mandatory while you are using '-s' option." << std::endl;
            LogFatal << "\"sample_cuts\" is an array which represent the accum level of each sample." << std::endl;
            LogFatal << "For example: [7,7,6]" << std::endl;
            throw std::logic_error("Could not find \"sample_cuts\" parameter.");
        }

        if(rebin_factor != -1){
            sampleRebin.resize(sampleCuts.size(), rebin_factor);
        }
        else if(j.find("sample_rebin") != j.end()){
            sampleRebin = j["sample_rebin"].get<std::vector<int>>();
            if(sampleRebin.size() != sampleCuts.size()){
                LogFatal << "Could not find \"sample_rebin\" array does not match \"sample_cuts\"'s one." << std::endl;
                throw std::logic_error("Invalid size of \"sample_rebin\" array.");
            }
            LogWarning << "Sample will be re-binned as follows:" << std::endl;
            LogWarning; GenericToolbox::printVector(sampleRebin);
        }

        std::string input_dir = GenericToolbox::getFolderPathFromFilePath(sampleDefFile) + "/";
        std::fstream configSampleDefinitionStream;

        configSampleDefinitionStream.open(sampleDefFile, std::ios::in);
        LogInfo << "Opening " << json_file << std::endl;
        if(not configSampleDefinitionStream.is_open())
        {
            LogError << "Unable to open JSON configure file." << std::endl;
            return 1;
        }
        json configJsonSampleDef;
        configSampleDefinitionStream >> configJsonSampleDef;

        std::string treeConverterFilePath = configJsonSampleDef["mc_file"];
        __treeConverterFile__       = TFile::Open((input_dir+treeConverterFilePath).c_str());
        __treeConverterRecoTTree__  = (TTree*) __treeConverterFile__->Get("selectedEvents");
        mapTreeConverterEntries();

        LogInfo << "Looking for defined samples..." << std::endl;

        for(const auto& configSample : configJsonSampleDef["samples"]){

            if(configSample["use_sample"] == false or configSample["cut_branch"] == -1){
                continue;
            }

            sampleList.emplace_back(SampleOpt());
            sampleList.back().cut_branch    = configSample["cut_branch"];
            sampleList.back().name          = configSample["name"];
            sampleList.back().detector      = configSample["detector"];
            sampleList.back().binning       = input_dir + configSample["binning"].get<std::string>();
            sampleList.back().use_sample    = configSample["use_sample"];
            if(configSample.find("additional_cuts") != configSample.end()) sampleList.back().additional_cuts = configSample["additional_cuts"].get<std::string>();
            cutFormulaList.emplace_back(nullptr);

            LogWarning << "Added sample: \"" << sampleList.back().name << "\"";

            if( not sampleList.back().additional_cuts.empty() ){
                LogWarning << "-> \"" << sampleList.back().additional_cuts << "\"";
                cutFormulaList.back() = new TTreeFormula(
                    Form("additional_cuts_%i", int(cutFormulaList.size())),
                    sampleList.back().additional_cuts.c_str(), __treeConverterRecoTTree__
                    );
                cutFormulaList.back()->SetTree(__treeConverterRecoTTree__);
                __treeConverterRecoTTree__->SetNotify(cutFormulaList.back()); // This is needed only for TChain.
            }
            LogWarning << std::endl;
        }

        LogInfo << "Re-binning samples for the covariance matrix..." << std::endl;
        for(size_t iSample = 0 ; iSample < sampleList.size() ; iSample++){
            cov_bin_manager.emplace_back(BinManager(sampleList[iSample].binning));
            if(not sampleRebin.empty()){
                cov_bin_manager.back().MergeBins(sampleRebin[iSample]);
            }
            LogInfo << GET_VAR_NAME_VALUE(iSample) << std::endl;
            cov_bin_manager.back().Print();
        }
    }
    else {
        // temp_cov_binning holds the binning text files for the different samples:
        std::map<std::string, std::string> temp_cov_binning = j["cov_sample_binning"];

        // Set length of cov_bin_manager vector to the number of samples:
        cov_bin_manager.resize(temp_cov_binning.size());

        // Loop over all the different samples:
        for( const auto& kv : temp_cov_binning ){
            //cov_bin_manager.at(std::stoi(kv.first)) = std::move(BinManager(kv.second));
            // Get the sample number and name of the binning file for the current sample:
            int sample_number = std::stoi(kv.first);
            std::string binning_file = kv.second;

            // Set up the binning from the binning text file and fill the cov_bin_manager vector with it:
            cov_bin_manager.at(sample_number) = BinManager(binning_file);
        }
    }

    // Number of samples:
    num_cov_samples = cov_bin_manager.size();

    // Set length of cov_bin_manager vector to the number of samples:
    cov_bin_manager.resize(num_cov_samples);

    // Loop over all the different samples:
    for( const auto& kv : temp_cov_binning )
    {
        //cov_bin_manager.at(std::stoi(kv.first)) = std::move(BinManager(kv.second));
        // Get the sample number and name of the binning file for the current sample:
        int sample_number = std::stoi(kv.first);
        std::string binning_file = kv.second;

        // Set up the binning from the binning text file and fill the cov_bin_manager vector with it:
        cov_bin_manager.at(sample_number) = BinManager(binning_file);
    }

    // Number of usable toys:
    unsigned int usable_toys = 0;

    // Vector which will hold the options for each file specified in the .json config file:
    std::vector<FileOptions> v_files;

    // Loop over all input Highland ROOT files:
    for(const auto& json_input_file : j["files"])
    {
        // Only consider the files which have the "use" key set to true:
        if(json_input_file["use"])
        {
            // If there is a "flist" key present in the .json config file, we add from the corresponding text file:
            if(json_input_file.find("flist") != json_input_file.end())
            {
                // Text file containing all files to be read in:
                std::ifstream in(json_input_file["flist"]);
                std::string filename;

                // Loop over lines of text file with the file list and add contents to the v_files vector:
                while(std::getline(in, filename))
                {
                    // Fill the FileOptions struct f with the file options in the .json config file:
                    FileOptions f_opt;
                    f_opt.fname_input = filename;
                    f_opt.tree_name   = json_input_file["tree_name"];
                    f_opt.detector    = json_input_file["detector"];
                    f_opt.num_toys    = int(json_input_file["num_toys"]);
                    f_opt.num_syst    = json_input_file["num_syst"];
                    f_opt.num_samples = json_input_file["num_samples"];
                    f_opt.cuts        = json_input_file["cuts"].get<std::vector<int>>();

                    if(not sampleDefFile.empty()){
                        for(size_t iSample = 0 ; iSample < sampleList.size() ; iSample++){
                            f_opt.samples.emplace(std::make_pair(iSample, sampleList[iSample].cut_branch));
                        }
                    }
                    else{
                        // temp_json is a map between sample numbers and selection branches in the Highland ROOT file (as specified in the .json config file):
                        std::map<std::string, std::vector<int>> temp_json = json_input_file["samples"];

                        // Loop over all the samples of the current file:
                        for(const auto& kv : temp_json)
                        {
                            // Number of the current sample:
                            const int sam = std::stoi(kv.first);

                            // If the number of the current sample is less than or equal to the total number of samples, we add an entry to the samples map of the FileOptions struct f (this is a map between sample number and a vector containing the selection branch numbers of the Highland ROOT file):
                            if(not sampleDefFile.empty() and sam <= num_cov_samples)
                                f_opt.samples.emplace(std::make_pair(sam, kv.second));

                                // If the nuber of the current sample is greater than the total number of samples, an error is thrown:
                            else
                            {
                                LogError << "Invalid sample number: " << sam << std::endl;
                                return 64;
                            }
                        }
                    }

                    // Get the mapping of variable names to branch numbers:
                    std::vector<std::map<std::string, std::vector<int>>> var_map_temp = json_input_file["variable_mapping"];
                    f_opt.variable_mapping = var_map_temp;

                    // Get the variable names and their total number:
                    unsigned int number_var_names = 0;
                    std::vector<std::string> variable_names_temp;
                    for(const auto& varmap : var_map_temp)
                    {
                        number_var_names += varmap.size();

                        for(const auto& kv : varmap)
                        {
                            variable_names_temp.emplace_back(kv.first);
                        }
                    }
                    f_opt.num_var_names = number_var_names;
                    f_opt.variable_names = variable_names_temp;

                    // Add the file options of the current file to the vector v_files:
                    v_files.emplace_back(f_opt);

                    LogWarning << GET_VAR_NAME_VALUE(f_opt.num_toys) << std::endl;

                    // As usable_toys was set to zero previously, this condition will be fulfilled and usable_toys will be set to the number of usable toys as specified in the .json config file:
                    if(f_opt.num_toys < usable_toys || usable_toys == 0)
                        usable_toys = f_opt.num_toys;
                }
            }

            // Otherwise we use the "fname_input" key to get the name of the file which is to be read in:
            else
            {

                // Fill the FileOptions struct f with the file options in the .json config file:
                FileOptions f_opt;
                f_opt.fname_input = json_input_file["fname_input"];
                f_opt.tree_name   = json_input_file["tree_name"];
                f_opt.detector    = json_input_file["detector"];
                f_opt.num_toys    = int(json_input_file["num_toys"]);
                f_opt.num_syst    = json_input_file["num_syst"];
                if(json_input_file.find("num_samples")  != json_input_file.end())
                    f_opt.num_samples = json_input_file["num_samples"];
                if(json_input_file.find("cuts")         != json_input_file.end())
                    f_opt.cuts        = json_input_file["cuts"].get<std::vector<int>>();

                if(not sampleDefFile.empty()){
                    for(size_t iSample = 0 ; iSample < sampleList.size() ; iSample++){
                        f_opt.samples[iSample] = {sampleCuts[iSample]};
                    }
                }
                else{
                    // temp_json is a map between sample numbers and selection branches in the Highland ROOT file (as specified in the .json config file):
                    std::map<std::string, std::vector<int>> temp_json = json_input_file["samples"];

                    // Loop over all the samples of the current file:
                    for(const auto& kv : temp_json)
                    {
                        // Number of the current sample:
                        const int sam = std::stoi(kv.first);

                        // If the number of the current sample is less than or equal to the total number of samples, we add an entry to the samples map of the FileOptions struct f (this is a map between sample number and a vector containing the selection branch numbers of the Highland ROOT file):
                        if(sam <= num_cov_samples){
                            f_opt.samples.emplace(std::make_pair(sam, kv.second));
                        }


                            // If the nuber of the current sample is greater than the total number of samples, an error is thrown:
                        else
                        {
                            LogError << "Invalid sample number: " << sam << std::endl;
                            return 64;
                        }
                    }
                }

                // Get the mapping of variable names to branch numbers:
                std::vector<std::map<std::string, std::vector<int>>> var_map_temp;
                if(json_input_file.find("variable_mapping") != json_input_file.end())
                    var_map_temp = json_input_file["variable_mapping"].get<std::vector<std::map<std::string, std::vector<int>>>>();
                f_opt.variable_mapping = var_map_temp;

                // Get the variable names and their total number:
                unsigned int number_var_names = 0;
                std::vector<std::string> variable_names_temp;
                for(const auto& varmap : var_map_temp)
                {
                    number_var_names += varmap.size();

                    for(const auto& kv : varmap)
                    {
                        variable_names_temp.emplace_back(kv.first);
                    }
                }
                f_opt.num_var_names = number_var_names;
                f_opt.variable_names = variable_names_temp;

                // Add the file options of the current file to the vector v_files:
                v_files.emplace_back(f_opt);

                LogWarning << GET_VAR_NAME_VALUE(f_opt.num_toys) << std::endl;

                // As usable_toys was set to zero previously, this condition will be fulfilled and usable_toys will be set to the number of usable toys as specified in the .json config file:
                if(f_opt.num_toys < usable_toys || usable_toys == 0)
                    usable_toys = f_opt.num_toys;
            }
        }
    }

    // Print some information about which options were set in the .json config file:
    LogInfo << "Output ROOT file: " << fname_output << std::endl
              << "Toy Weight Cut: " << weight_cut << std::endl
              << "Calculating Covariance: " << std::boolalpha << do_covariance << std::endl;

    // Print the covariance variables as specified in the .json config file:
    LogInfo << "Covariance Variables: ";
    for(const auto& var : var_names)
        std::cout << var << " ";
    std::cout << std::endl;

    // If "projection" was set to true in the .json config file (this means plots are saved with axis as kinematic variable instead of bin number), we get the location (stored in var_plot) of the "plot_variable" in the "var_names" array:
    int var_plot = -1;
    if(do_projection)
    {
        auto it  = std::find(var_names.begin(), var_names.end(), variable_plot);
        var_plot = std::distance(var_names.begin(), it);
    }

    // Now starting with the histogram initialization and printing the number of toys which will be used:
    LogInfo << "Initalizing histograms." << std::endl;
    LogInfo << "Using " << usable_toys << " toys." << std::endl;

    // All new histograms will automatically activate the storage of the sum of squares of errors:
    TH1::SetDefaultSumw2();

    // Vector (will have same length as number of samples) which will hold vectors of ROOT TH1F histograms (will have length equal to the number of toys):
    std::vector<std::vector<TH1F>> v_hists;

    // Vectors (will have same length as number of samples) of TH1F ROOT histograms holding the averages and the MC statistical errors:
    std::vector<TH1F> v_avg;
    std::vector<TH1F> v_mc_stat;

    // Loop over all samples:
    for(int i = 0; i < cov_bin_manager.size(); ++i)
    {
        // Get the binning of the current sample:
        BinManager bm   = cov_bin_manager.at(i);

        // Number of bins of current sample:
        const int nbins = bm.GetNbins();

        // Vector of histograms for current sample which will be added to v_hists at the end of this for loop:
        std::vector<TH1F> v_temp;

        // Loop over all toys:
        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            // Name of current histogram (e.g., cov_sample2_toy345):
            std::stringstream ss;
            ss << "cov_sample" << i << "_toy" << t;

            // If do_projection is true, plots are saved with axis as kinematic variable (specified in .json config file, e.g., selmu_mom) instead of bin number:
            if(do_projection)
            {
                // Vector of bin edges of the plot_variable (which was specified in the .json config file, e.g., selmu_mom):
                std::vector<double> v_bins = bm.GetBinVector(var_plot);

                // Create a new ROOT TH1F histogram (name and title are ss, number of bins is the size of v_bins minus 1, and the binning is given by bin edges of the plot_variable) and add it to the v_temp vector of the current sample:
                v_temp.emplace_back(
                    TH1F(ss.str().c_str(), ss.str().c_str(), v_bins.size() - 1, &v_bins[0]));

                // If it is the very first toy, we create new ROOT TH1F histograms for holding the averages and the MC statistical errors and add them to the v_avg and v_mc_stat vectors (these will have the same length as the number of samples):
                if(t == 0)
                {
                    ss.str("");
                    ss << "cov_sample" << i << "_avg";
                    v_avg.emplace_back(
                        TH1F(ss.str().c_str(), ss.str().c_str(), v_bins.size() - 1, &v_bins[0]));

                    ss.str("");
                    ss << "cov_sample" << i << "_mc_stat";
                    v_mc_stat.emplace_back(
                        TH1F(ss.str().c_str(), ss.str().c_str(), v_bins.size() - 1, &v_bins[0]));
                }
            }

            // If do_projection is false, plots are saved with axis as bin number (instead of kinematic variable):
            else
            {
                // Create a new ROOT TH1F histogram (name and title are ss, number of bins is nbins, first bin is zero, and last bin is equal to the number of bins (the bins will be equally spaced)) and add it to the v_temp vector of the current sample:
                v_temp.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));

                // If it is the very first toy, we create new ROOT TH1F histograms for holding the averages and the MC statistical errors and add them to the v_avg and v_mc_stat vectors (these will have the same length as the number of samples):
                if(t == 0)
                {
                    ss.str("");
                    ss << "cov_sample" << i << "_avg";
                    v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));

                    ss.str("");
                    ss << "cov_sample" << i << "_mc_stat";
                    v_mc_stat.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));
                }
            }
        }

        // Add the vector with all the histograms (for all the toys) for the current sample to v_hists:
        v_hists.emplace_back(v_temp);
    }

    // Print information, that the initialization of the histograms was has finished and we can move to reading the events from the ROOT files:
    LogInfo << "Finished initializing histograms" << std::endl
              << "Reading events from files..." << std::endl;

    // Loop over all files specified in the .json config file:
    for(const auto& file_ : v_files)
    {
        // Declare some variables that will hold the values that we need from the input ROOT files:
        int NTOYS = 0;
        int accum_level[file_.num_toys][2][file_.num_samples]; // 2 stands for FGD1 and 2
        float hist_variables[file_.num_var_names][file_.num_toys];
        float weight_syst_total_noflux[file_.num_toys];
        float weight_syst[file_.num_toys][file_.num_syst];

        int accum_level_mc[file_.num_toys][2][file_.num_samples];
        float hist_variables_mc[file_.num_var_names];
        float weight_syst_total_noflux_mc;

        // Print some information about which file is opened, which tree is read in, the number of toys and systematics, and which selection branches are mapped to which selection samples:
        LogInfo << "Opening file: " << file_.fname_input << std::endl
                  << "Reading tree: " << file_.tree_name << std::endl
                  << "Num Toys: " << file_.num_toys << std::endl
                  << "Num Syst: " << file_.num_syst << std::endl;

        LogInfo << "Branch to Sample mapping:" << std::endl;
        for(const auto& kv : file_.samples)
        {
            LogInfo << "Sample " << kv.first << ": ";
            for(const auto& b : kv.second)
                std::cout << b << " ";
            std::cout << std::endl;
        }

        // Open current ROOT input file and access the default tree and the one specified in the .json config file (e.g, all_syst):
        TFile* file_input = TFile::Open(file_.fname_input.c_str(), "READ");
        auto* tree_event = (TTree*)file_input->Get(file_.tree_name.c_str());
        auto* tree_default = (TTree*)file_input->Get("default");

        __currentInputFile__ = file_input; // globals
        if(__treeConverterRecoTTree__ != nullptr) buildTreeSyncCache();

        // Set the branch addresses for the selected tree to the previously declared variables:
        tree_event->SetBranchAddress("NTOYS", &NTOYS);
        tree_event->SetBranchAddress("accum_level", accum_level);
        tree_event->SetBranchAddress("weight_syst", weight_syst);
        tree_event->SetBranchAddress("weight_syst_total", weight_syst_total_noflux);

        // Loop over all the variable names specified in the .json config file (e.g., selmu_mom, selmu_mom_range_oarecon and selmu_costheta):
        for(unsigned int i = 0; i < file_.num_var_names; ++i)
        {
            // Set the branch address of the current variable of the hist_variables (declared above) array to the variable name:
            tree_event->SetBranchAddress(file_.variable_names[i].c_str(), hist_variables[i]);
        }

        // Set the branch addresses for the default tree to the previously declared variables:
        tree_default->SetBranchAddress("accum_level", accum_level_mc);
        tree_default->SetBranchAddress("weight_syst_total", &weight_syst_total_noflux_mc);

        // Loop over all the variable names specified in the .json config file (e.g., selmu_mom, selmu_mom_range_oarecon and selmu_costheta):
        for(unsigned int i = 0; i < file_.num_var_names; ++i)
        {
            // Set the branch address of the current variable of the hist_variables_mc (declared above) array to the variable name:
            tree_default->SetBranchAddress(file_.variable_names[i].c_str(), &hist_variables_mc[i]);
        }

        // Numbers which will hold the number of rejected events, number of events which passed all the cuts, and total number of events:
        unsigned int rejected_weights = 0;
        unsigned int total_weights    = 0;
        unsigned int num_events = tree_event->GetEntries();

        // Print out total number of events:
        LogInfo << "Number of events: " << num_events << std::endl;

        // Loop over all events:
        for(unsigned int i = 0; i < num_events; ++i)
        {
            // Get the ith event:
            tree_event->GetEntry(i);
            __treeConverterRecoTTree__->GetEntry(__currentFlatToTreeConvEntryMapping__[i]);

            // If the number of toys from the input ROOT file does not match the number of toys specified in the .json config file, an error message is printed:
            if(NTOYS != file_.num_toys)
                LogError << "Incorrect number of toys specified!" << std::endl;

            //  Update progress bar every 2000 events (or if it is the last event):
            if(i % 2000 == 0 || i == (num_events - 1))
                pbar.Print(i, num_events - 1);

            // Loop over all toys:
            for(unsigned int i_toy = 0; i_toy < usable_toys; ++i_toy)
            {
                // Loop over all selection samples:
                std::map<int, std::vector<int>> my_samples = file_.samples;
                for(const auto& my_sample : my_samples)
                {
                    // sample_id is the sample number:
                    unsigned int sample_id = my_sample.first;

                    for(unsigned int i_FGD = 0 ; i_FGD < 2 ; i_FGD++){

                        __currentFGD__ = i_FGD;

                        // Loop over all selection branches in current selection sample:
                        for(const auto& branch : my_sample.second)
                        {

                            // Only consider events that passed the selection (accum_level is higher than the given cut for this branch):
                            if(accum_level[i_toy][i_FGD][branch] > file_.cuts[sample_id])
                            {

                                if(not sampleList.empty() and not doesEntryPassAdditionalCut(tree_event, sample_id)){
                                    continue;
                                }

                                // Get the names and locations (indices) of the kinematic variables for the current branch:
                                std::vector<std::string> variable_names_current_branch;
                                std::vector<int> variable_loc_current_branch;

                                // Loop over the number of kinematic variables:
                                for(unsigned int Var = 0; Var < nvars; ++Var)
                                {
                                    // Location of the current variable, will be updated below:
                                    int loc_current_var = 0;

                                    // Loop over the number of variables used for the current kinematic variable:
                                    for(const auto& kv_ : file_.variable_mapping[Var])
                                    {
                                        // If the current branch is in the vector of branch numbers for current variable, we add the current variable name to variable_names_current_branch:
                                        if(std::find(kv_.second.begin(), kv_.second.end(), branch) != kv_.second.end())
                                        {
                                            variable_names_current_branch.push_back(kv_.first);
                                            variable_loc_current_branch.push_back(loc_current_var);
                                            break;
                                        }

                                        // Update location (index) of current variable:
                                        ++loc_current_var;
                                    }
                                }

                                // idx is the bin number for the current event (will be updated below) in the ROOT histograms which have been initialized:
                                int idx = -1;

                                // If do_projection is true, plots are saved with axis as kinematic variable (specified in .json config file, e.g., selmu_mom) instead of bin number. In this case idx will be the value of the plot variable (e.g., selmu_mom) for the current toy of this event:
                                if(do_projection)
                                    idx = int(hist_variables[variable_loc_current_branch[var_plot]][i_toy]);

                                    // If do_projection is false, plots are saved with axis as bin number (instead of kinematic variable). In this case idx will be the index of the bin that this event/toy falls in:
                                else
                                {
                                    // vars will hold the values of the kinematic variables for the current event and toy:
                                    std::vector<double> vars;

                                    // Counts number of variables used for previous kinematic variables so that the correct index of hist_variables is used down below:
                                    int count = 0;

                                    // Loop over the kinematic variables (e.g., muon angle and muon momentum) and add the value for the current event and toy to the vars vector:
                                    for(unsigned int v = 0; v < nvars; ++v)
                                    {
                                        int current_index = variable_loc_current_branch[v] + count;
                                        vars.push_back(hist_variables[current_index][i_toy]);
                                        count += file_.variable_mapping[v].size();
                                    }

                                    // We get the index of the bin for this event and this sample (as specified in the cov_sample_binning files) and set idx equal to it:
                                    idx = cov_bin_manager[sample_id].GetBinIndex(vars);
                                }

                                // If "single_syst" has been set to true in the .json config file, the weight for the current event and toy will be the weight given by this single systematic (which needs to be given the correct index in the .json config file). Otherwise the weight will be the total weight for this event and toy:
                                float weight = do_single_syst ? weight_syst[i_toy][syst_idx]
                                                              : weight_syst_total_noflux[i_toy];

                                // If the weight for this event and toy is between zero and the weight cut set in the .json config file, we fill the ROOT histogram for the current sample and toy with the weight of the event in the bin that this event falls in. We also fill the average histogram for the current sample (for all toys) with the weight divided by the number of toys:
                                if(weight > 0.0 && weight < weight_cut)
                                {
                                    v_hists[sample_id][i_toy].Fill(idx, weight);
                                    v_avg[sample_id].Fill(idx, weight / double(file_.num_toys));
                                }

                                    // If the weight for this event and toy is less than zero or greater than the weight cut, we ignore the event and increase the rejected_weights number by 1:
                                else
                                    rejected_weights++;

                                // total_weights is the number of events that passed all selection cuts. This is increased by 1:
                                total_weights++;

                                // If the event/toy passes the selection cuts for the current branch, we break and move on:
                                break;
                            }
                        }

                    }


                }
            }
        }

        // We now move on to reading out the default tree:
        LogInfo << "Reading default events..." << std::endl;

        // Get the number of events:
        num_events = tree_default->GetEntries();

        // Loop over all events:
        for(unsigned int i = 0; i < num_events; ++i)
        {
            // Get the ith event from the default tree:
            tree_default->GetEntry(i);

            // Loop over all selection samples:
            for(const auto& kv : file_.samples)
            {
                // s is the sample number:
                unsigned int s = kv.first;

                for(unsigned int i_FGD = 0 ; i_FGD < 2 ; i_FGD++){
                    // Loop over all selection branches in current selection sample:
                    for(const auto& branch : kv.second)
                    {
                        // Only consider events that passed the selection (accum_level is higher than the given cut for this branch):
                        if(accum_level_mc[0][i_FGD][branch] > file_.cuts[branch])
                        {
                            // Get the names and locations (indices) of the kinematic variables for the current branch:
                            std::vector<std::string> variable_names_current_branch;
                            std::vector<int> variable_loc_current_branch;

                            // Loop over the number of kinematic variables:
                            for(unsigned int Var = 0; Var < nvars; ++Var)
                            {
                                // Location of the current variable, will be updated below:
                                int loc_current_var = 0;

                                // Loop over the number of variables used for the current kinematic variable:
                                for(const auto& kv_ : file_.variable_mapping[Var])
                                {
                                    // If the current branch is in the vector of branch numbers for current variable, we add the current variable name to variable_names_current_branch:
                                    if(std::find(kv_.second.begin(), kv_.second.end(), branch) != kv_.second.end())
                                    {
                                        variable_names_current_branch.push_back(kv_.first);
                                        variable_loc_current_branch.push_back(loc_current_var);
                                        break;
                                    }

                                    // Update location (index) of current variable:
                                    ++loc_current_var;
                                }
                            }

                            // idx is the bin number for the current event (will be updated below) in the ROOT histograms which have been initialized:
                            int idx = -1;

                            // If do_projection is true, plots are saved with axis as kinematic variable (specified in .json config file, e.g., selmu_mom) instead of bin number. In this case idx will be the value of the current plot variable (e.g., selmu_mom) for the current event:
                            if(do_projection)
                                idx = int(hist_variables_mc[variable_loc_current_branch[var_plot]]);

                                // If do_projection is false, plots are saved with axis as bin number (instead of kinematic variable). In this case idx will be the index of the bin that this event falls in:
                            else
                            {
                                // vars will hold the values of the kinematic variables for the current event:
                                std::vector<double> vars;

                                // Counts number of variables used for previous kinematic variables so that the correct index of hist_variables is used down below:
                                int count = 0;

                                // Loop over the kinematic variables (e.g., muon angle and muon momentum) and add the value for the current event and toy to the vars vector:
                                for(unsigned int v = 0; v < nvars; ++v)
                                {
                                    int current_index = variable_loc_current_branch[v] + count;
                                    vars.push_back(hist_variables_mc[current_index]);
                                    count += file_.variable_mapping[v].size();
                                }

                                // We get the index of the bin for this event and this sample (as specified in the cov_sample_binning files) and set idx equal to it:
                                idx = cov_bin_manager[s].GetBinIndex(vars);
                            }

                            // The weight of the current event is given by weight_syst_total_noflux_mc:
                            float weight = weight_syst_total_noflux_mc;

                            // We fill the ROOT histogram for the current sample with the weight of the event in the bin that this event falls in:
                            v_mc_stat[s].Fill(idx, weight);

                            // If the event passes the selection cuts for the current branch, we break and move on:
                            break;
                        }
                    }
                }

            }
        }

        // Print some information about how many events passed all the selection cuts (total_weights), how many had a weight less than zero or higher than the weight cut (rejected_weights), and their fraction:
        double reject_fraction = (rejected_weights * 1.0) / total_weights;
        LogInfo << "Finished processing events." << std::endl;
        LogInfo << "Total weights: " << total_weights << std::endl;
        LogInfo << "Rejected weights: " << rejected_weights << std::endl;
        LogInfo << "Rejected fraction: " << reject_fraction << std::endl;

        // Close the current ROOT input file and move on to the next one (it this was not the last one):
        file_input->Close();
    }

    // Size of coavariance and correlation matrices (will be updated below):
    unsigned int num_elements = 0;

    // initialize the covariance and correlation matrices (symmetric ROOT matrices):
    TMatrixTSym<double> cov_mat(num_elements);
    TMatrixTSym<double> cor_mat(num_elements);

    // Vector which will be filled with the MC statistical errors:
    std::vector<float> v_mc_error;
    TH1D* data_histogram = nullptr;

    // If the covariance key is set to true in the .json config file, we compute the covariance and correlation matrices below:
    if(do_covariance)
    {
        LogInfo << "Calculating covariance matrix." << std::endl;

        // Vector which will later be filled with the bin content (sum of all weights of events that fall into this bin) for each sample and each toy:
        std::vector<std::vector<float>> v_toys;

        // Loop over all toys:
        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            // Vector which will later be filled with the bin content (sum of all weights of events that fall into this bin) for each sample for current toy t:
            std::vector<float> i_toy;

            // Loop over all samples:
            for(int s = 0; s < cov_bin_manager.size(); ++s)
            {
                // Number of bins in current sample s:
                const unsigned int nbins = cov_bin_manager[s].GetNbins();

                // Loop over all bins in current sample:
                for(unsigned int b = 0; b < nbins; ++b)
                    {
                        // Get the bin content of bin b (sum of all weights of events that fall into this bin) for current sample s and toy t, and add this to i_toy vector:
                        i_toy.emplace_back(v_hists[s][t].GetBinContent(b + 1));
                    }

            }

            // Add the vector with the bin content for all samples for current toy t to the v_toys vector:
            v_toys.emplace_back(i_toy);
        }

        // Loop over all samples:
        for(int s = 0; s < cov_bin_manager.size(); ++s)
        {
            // Number of bins in current sample s:
            const unsigned int nbins = cov_bin_manager[s].GetNbins();

            // Array of the bin content (sum of all weights of events that fall into this bin) for current sample s (v_mc_stat was filled from the default tree):
            float* w   = v_mc_stat[s].GetArray();

            // Array of the sum of squares of the bin content for current sample s (v_mc_stat was filled from the default tree):
            double* w2 = v_mc_stat[s].GetSumw2()->GetArray();
            //std::cout << "Sample " << s << std::endl;

            // Loop over all bins in current sample:
            for(unsigned int b = 0; b < nbins; ++b)
            {
                //std::cout << "Bin : " << w[b+1] << std::endl;
                //std::cout << "W2  : " << w2[b+1] << std::endl;

                // Compute the relative error for current sample s and current bin b:
                float rel_error = float(w2[b+1]) / (w[b+1] * w[b+1]);

                // Add the relative error for current sample and bin to the v_mc_error vector:
                v_mc_error.emplace_back(rel_error);
            }
        }

        // Print information about the number of toys we are using:
        LogInfo << "Using " << usable_toys << " toys." << std::endl;

        // Number of elements is the sum of the number of bins in each sample:
        num_elements = v_toys.at(0).size();

        // Vector which will be filled with the mean values over all toys of the bin content for each bin in each sample:
        std::vector<float> v_mean(num_elements, 0.0);

        // The covariance and correlation matrices are resized to square matrices with the number of rows and columns being equal to the number of elements:
        cov_mat.ResizeTo(num_elements, num_elements);
        cor_mat.ResizeTo(num_elements, num_elements);

        // All the matrix elements for the covariance and correlation matrices are set to zero:
        cov_mat.Zero();
        cor_mat.Zero();

        // Loop over all toys:
        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            // Loop over all bins in all samples:
            for(unsigned int i = 0; i < num_elements; ++i)
            {
                // Compute the mean over all toys of the bin content for each bin in each sample:
                v_mean[i] += v_toys[t][i] / (1.0 * usable_toys);
            }
        }


        // Loop over all toys:
        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            // Loop over all bins in all samples:
            for(unsigned int i = 0; i < num_elements; ++i)
            {
                // Second loop over all bins in all samples:
                for(unsigned int j_ = 0; j_ < num_elements; ++j_)
                {
                    // Only compute the matrix element, if the denominators in the expression below are both not zero:
                    if(v_mean[i] != 0 && v_mean[j_] != 0)
                    {
                        // Compute current matrix element of the covariance matrix:
                        cov_mat(i, j_) += (1.0 - v_toys[t][i] / v_mean[i])
                                         * (1.0 - v_toys[t][j_] / v_mean[j_]) / (1.0 * usable_toys);
                    }
                }
            }
        }

        // If the "mc_stat_error" key was set to true in the .json config file, we add the MC statistical error to the diagonal entries of the covariance matrix:
        if(do_mc_stat)
        {
            LogInfo << "Adding MC stat error to covariance." << std::endl;

            // Loop over all bins in all samples:
            for(unsigned int i = 0; i < num_elements; ++i)
            {
                // Add the MC statistical errors to the diagonal entries of the covariance matrix:
                cov_mat(i, i) += v_mc_error[i];
            }
        }

        // If the diagonal entries in the covariance matrix are less than zero, we set them to one:
        for(unsigned int i = 0; i < num_elements; ++i)
        {
            if(cov_mat(i, i) <= 0.0)
                cov_mat(i, i) = 1.0;
        }

        // Loop over all bins in all samples
        for(unsigned int i = 0; i < num_elements; ++i)
        {
            // Second loop over all bins in all samples:
            for(unsigned int j_ = 0; j_ < num_elements; ++j_)
            {
                // Get the diagonal entries of the covariance matrix:
                double bin_i  = cov_mat(i, i);
                double bin_j  = cov_mat(j_, j_);

                // Compute current matrix element of the correlation matrix:
                cor_mat(i, j_) = cov_mat(i, j_) / std::sqrt(bin_i * bin_j);

                // If the current matrix element is a nan, we set it to zero:
                if(std::isnan(cor_mat(i, j_)))
                    cor_mat(i, j_) = 0;
            }
        }


        std::vector<double> v_mean_double(v_mean.begin(), v_mean.end());
        auto* bin_contents = GenericToolbox::convertStdVectorToTVectorD(v_mean_double);
        auto* diagonal_values = new TVectorD(num_elements);
        for(int i_diag = 0 ; i_diag < num_elements ; i_diag++){
            (*diagonal_values)[i_diag] = TMath::Sqrt(cov_mat(i_diag,i_diag)*(*bin_contents)[i_diag]);
        }
        data_histogram = GenericToolbox::convertTVectorDtoTH1D(
            bin_contents,
            "MC_data_and_detector_errors", "Counts", "Bin #",
            diagonal_values);

    }

    LogInfo << "Saving to output file." << std::endl;

    // Create the output file. If it already exists, it will be overwritten:
    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output->cd();

    data_histogram->Write();

    // Control the information printed in the statistics box of the produced plots:
    gStyle->SetOptStat(0);

    // Loop over all samples:
    for(int s = 0; s < cov_bin_manager.size(); ++s)
    {
        // Name and title of the ROOT TCanvas (s is the sample number):
        std::stringstream ss;
        ss << "cov_sample" << s;

        // Create a TCanvas with name and title equal to ss and pixel size of 1200x900:
        TCanvas c(ss.str().c_str(), ss.str().c_str(), 1200, 900);

        // Draw the average over all toys for the current sample:
        v_avg[s].Draw("axis");

        // Loop over all toys:
        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            v_hists[s][t].SetLineColor(kRed);
            if(do_projection)
                v_hists[s][t].Scale(1, "width");
            v_hists[s][t].Draw("hist same");
        }

        v_avg[s].SetLineColor(kBlack);
        v_avg[s].SetLineWidth(2);
        if(do_projection)
            v_avg[s].Scale(1, "width");
        v_avg[s].GetYaxis()->SetRangeUser(0, v_avg[s].GetMaximum() * 1.50);
        v_avg[s].Draw("hist same");
        c.Write(ss.str().c_str());

        // If the "pdf_print" key has been set to true in the .json config file, the plots are saved as PDFs:
        if(do_print)
            c.Print(std::string(ss.str() + ".pdf").c_str());
    }

    // If the covariance key is set to true in the .json config file, we add the covariance and correlation matrices to the output ROOT file:
    if(do_covariance)
    {
        cov_mat.Write(cov_mat_name.c_str());
        cor_mat.Write(cor_mat_name.c_str());
    }

    if(do_mc_stat)
    {
        TVectorT<float> v_mc_root(v_mc_error.size(), v_mc_error.data());
        v_mc_root.Write("mc_stat_error");
    }

    // Close the output file:
    file_output->Close();

    LogInfo << "Finished." << std::endl;

    // Print Arigatou Gozaimashita with Rainbowtext :)
    LogInfo << color::RainbowText("\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01")
              << std::endl;

    return 0;
}


bool doesEntryPassAdditionalCut(TTree* tree_, int sampleId_){

    if(sampleList[sampleId_].additional_cuts.empty()) return true;

    __treeConverterRecoTTree__->SetNotify(cutFormulaList[sampleId_]);
    bool doEventPassCut = true;
    for(int jInstance = 0; jInstance < cutFormulaList[sampleId_]->GetNdata(); jInstance++) {
        if ( cutFormulaList[sampleId_]->EvalInstance(jInstance) == 0 ) {
            doEventPassCut = false;
            break;
        }
    }

    return doEventPassCut;

}
void initCutDictionnary(){

    __cutDictionnary__["beammode"] = [](TTree* tree){
        double run = tree->GetLeaf("sRun")->GetValue(0);

        int beam_mode = 0;
        if(int(run/1E7) == 9){
            beam_mode = 1;
        }
        else if(int(run/1E7) == 8){
            beam_mode = -1;
        }
        else {
            beam_mode = 0;
        }
        return double(beam_mode);
    };

    __cutDictionnary__["fgd_reco"] = [](TTree* tree){
        return __currentFGD__;
    };

    __cutDictionnary__["analysis"] = [](TTree* tree){

        double analysis = 0;

        auto splitFileName = GenericToolbox::splitString(GenericToolbox::getFileNameFromFilePath(__currentInputFile__->GetName()), "_");
        if(splitFileName.size() > 2){
            if(splitFileName[1] == "NumuCCMultiPiAnalysis"){
                analysis = 1;
            }
            else if(splitFileName[1] == "AntiNumuCCMultiPiAnalysis"){
                analysis = -1;
            }
        }

        return analysis;
    };

}
bool buildTreeSyncCache(){

    if(__treeConvEntryToVertexIDSplit__.empty()) mapTreeConverterEntries();

    LogInfo << "Building tree sync cache..." << std::endl;

    // genWeights tree (same number of events as __flattree__)
    int nbGenWeightsEntries = __currentInputTree__->GetEntries();

    Int_t sTrueVertexID;
    Int_t sRun;
    Int_t sSubRun;

    __currentInputTree__->SetBranchStatus("*", false);

    __currentInputTree__->SetBranchStatus("sTrueVertexID", true);
    __currentInputTree__->SetBranchStatus("sRun", true);
    __currentInputTree__->SetBranchStatus("sSubRun", true);

//    __currentFlatTree__->SetBranchAddress("sTrueVertexID[0]", &sTrueVertexID); // Can't do that
    __currentInputTree__->SetBranchAddress("sRun", &sRun);
    __currentInputTree__->SetBranchAddress("sSubRun", &sSubRun);

    __currentFlatToTreeConvEntryMapping__.clear();

    int nbMatches = 0, nbMissing = 0;
    for(int iGenWeightsEntry = 0 ; iGenWeightsEntry < nbGenWeightsEntries; iGenWeightsEntry++){ // genWeights
        __currentInputTree__->GetEntry(iGenWeightsEntry);
        __currentFlatToTreeConvEntryMapping__[iGenWeightsEntry] = -1; // reset

        int tcEntry = -1;
        int tcVertexID = -1;
        for(const auto& runMapPair : __treeConvEntryToVertexIDSplit__){
            if(runMapPair.first != sRun) continue;
            for(const auto& subrunMapPair : runMapPair.second){
                if(subrunMapPair.first != sSubRun) continue;

                sTrueVertexID = __currentInputTree__->GetLeaf("sTrueVertexID")->GetValue(0);

                for(const auto& tcEntryToVertexID : subrunMapPair.second){
                    tcEntry = tcEntryToVertexID.first;
                    tcVertexID = tcEntryToVertexID.second;

                    // Looking for the sTrueVertexID in TC
                    if(tcVertexID != sTrueVertexID ) {
                        continue;
                    }
                    else{
                        // Check if the corresponding run/subrun matches
                        __currentFlatToTreeConvEntryMapping__[iGenWeightsEntry] = tcEntry;
                        break;

                    }
                }
            }
        }
        if(__currentFlatToTreeConvEntryMapping__[iGenWeightsEntry] == -1){
            nbMissing++;
        }
        else {
            nbMatches++;
        }
    }

    LogInfo << "Tree synchronization has: " << nbMatches << " matches, " << nbMissing << " miss." << std::endl;

    __currentInputTree__->ResetBranchAddress(__currentInputTree__->GetBranch("sTrueVertexID"));
    __currentInputTree__->ResetBranchAddress(__currentInputTree__->GetBranch("sRun"));
    __currentInputTree__->ResetBranchAddress(__currentInputTree__->GetBranch("sSubRun"));

//    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("run"));
//    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("subrun"));

    __currentInputTree__->SetBranchStatus("*", true);

    if(nbMatches == 0){
        LogError << "Could not sync genWeights file and TreeConverter." << std::endl;
        return false;
    }
    return true;

}
void mapTreeConverterEntries(){

    LogInfo << "Mapping Tree Converter File..." << std::endl;

    Int_t vertexID;
    int run;
    int subrun;
    __treeConverterRecoTTree__->SetBranchAddress("vertexID", &vertexID);
    __treeConverterRecoTTree__->SetBranchAddress("run", &run);
    __treeConverterRecoTTree__->SetBranchAddress("subrun", &subrun);

    std::string loadTitle = LogInfo.getPrefixString() + "Mapping TC entries...";
    for(int iEntry = 0 ; iEntry < __treeConverterRecoTTree__->GetEntries(); iEntry++){
        GenericToolbox::displayProgressBar(iEntry, __treeConverterRecoTTree__->GetEntries(), loadTitle);
        __treeConverterRecoTTree__->GetEntry(iEntry);
        __treeConvEntryToVertexIDSplit__[run][subrun][iEntry] = vertexID;
    }
    __treeConverterRecoTTree__->ResetBranchAddress(__treeConverterRecoTTree__->GetBranch("vertexID"));

}