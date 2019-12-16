#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TStyle.h"
#include "TTree.h"
#include "TVectorT.h"

#include "json.hpp"
using json = nlohmann::json;

#include "BinManager.hh"
#include "ColorOutput.hh"
#include "ProgressBar.hh"

// Structure that holds the options and the binning for a file specified in the .json config file:
struct FileOptions
{
    std::string fname_input;
    std::string tree_name;
    std::string detector;
    unsigned int num_samples;
    unsigned int num_toys;
    unsigned int num_syst;
    std::vector<int> cuts;
    std::map<int, std::vector<int>> samples;
    std::vector<BinManager> bin_manager;
};

int main(int argc, char** argv)
{
    // Define colors and strings for info and error messages:
    const std::string TAG = color::GREEN_STR + "[xsDetVariation]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    // Print welcome message:
    std::cout << "--------------------------------------------------------\n"
              << TAG << color::RainbowText("Welcome to the Super-xsLLh Detector Variation Interface.\n")
              << TAG << color::RainbowText("Initializing the variation machinery...") << std::endl;

    // Progress bar for reading in events from the ROOT file:
    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    // .json config file that will be parsed from the command line:
    std::string json_file;

    // Initialize json_file with the name parsed from the command line using -j. Print USAGE and exit when -h is used:
    char option;
    while((option = getopt(argc, argv, "j:h")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n";
            default:
                return 0;
        }
    }

    // Read in the .json config file:
    std::fstream f;
    f.open(json_file, std::ios::in);
    std::cout << TAG << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        std::cout << ERR << "Unable to open JSON configure file." << std::endl;
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
    std::string variable_plot = j["plot_variable"];
    std::string cov_mat_name  = j["covariance_name"];
    std::string cor_mat_name  = j["correlation_name"];

    std::vector<std::string> var_names = j["var_names"].get<std::vector<std::string>>();
    const int nvars = var_names.size();

    // cov_bin_manager will hold the binnings of the different samples (the length of this vector will be equal to the number of samples):
    std::vector<BinManager> cov_bin_manager;

    // temp_cov_binning holds the binning text files for the different samples:
    std::map<std::string, std::string> temp_cov_binning = j["cov_sample_binning"];

    // Number of samples:
    const unsigned int num_cov_samples = temp_cov_binning.size();

    // Set length of cov_bin_manager vector to the number of samples:
    cov_bin_manager.resize(num_cov_samples);

    // Loop over all the different samples:
    for(const auto& kv : temp_cov_binning)
    {
        //cov_bin_manager.at(std::stoi(kv.first)) = std::move(BinManager(kv.second));
        // Get the sample number and name of the binning file for the current sample:
        int sample_number = std::stoi(kv.first);
        std::string binning_file = kv.second;

        // Set up the binning from the binning text file and fill the cov_bin_manager vector with it:
        cov_bin_manager.at(sample_number) = std::move(BinManager(binning_file));        
    }

    // Number of usable toys:
    unsigned int usable_toys = 0;

    // Vector which will hold the options for each file specified in the .json config file:
    std::vector<FileOptions> v_files;

    // Loop over all input Highland ROOT files:
    for(const auto& file : j["files"])
    {
        // Only consider the files which have the "use" key set to true:
        if(file["use"])
        {
            // If there is a "flist" key present in the .json config file, we add from the corresponding text file:
            if(file.find("flist") != file.end())
            {
                // Text file containing all files to be read in:
                std::ifstream in(file["flist"]);
                std::string filename;

                // Loop over lines of text file with the file list and add contents to the v_files vector:
                while(std::getline(in, filename))
                {
                    // Fill the FileOptions struct f with the file options in the .json config file:
                    FileOptions f;
                    f.fname_input = filename;
                    f.tree_name   = file["tree_name"];
                    f.detector    = file["detector"];
                    f.num_toys    = file["num_toys"];
                    f.num_syst    = file["num_syst"];
                    f.num_samples = file["num_samples"];
                    f.cuts        = file["cuts"].get<std::vector<int>>();

                    // temp_json is a map between sample numbers and selection branches in the Highland ROOT file (as specified in the .json config file):
                    std::map<std::string, std::vector<int>> temp_json = file["samples"];

                    // Loop over all the samples of the current file:
                    for(const auto& kv : temp_json)
                    {
                        // Number of the current sample:
                        const int sam = std::stoi(kv.first);

                        // If the number of the current sample is less than or equal to the total number of samples, we add an entry to the samples map of the FileOptions struct f (this is a map between sample number and a vector containing the selection branch numbers of the Highland ROOT file):
                        if(sam <= num_cov_samples)
                            f.samples.emplace(std::make_pair(sam, kv.second));

                        // If the nuber of the current sample is greater than the total number of samples, an error is thrown:
                        else
                        {
                            std::cout << ERR << "Invalid sample number: " << sam << std::endl;
                            return 64;
                        }
                    }

                    // Add the file options of the current file to the vector v_files:
                    v_files.emplace_back(f);

                    // As usable_toys was set to zero previously, this condition will be fulfilled and usable_toys will be set to the number of usable toys as specified in the .json config file:
                    if(f.num_toys < usable_toys || usable_toys == 0)
                        usable_toys = f.num_toys;
                }
            }

            // Otherwise we use the "fname_input" key to get the name of the file which is to be read in:
            else
            {
                // Fill the FileOptions struct f with the file options in the .json config file:
                FileOptions f;
                f.fname_input = file["fname_input"];
                f.tree_name   = file["tree_name"];
                f.detector    = file["detector"];
                f.num_toys    = file["num_toys"];
                f.num_syst    = file["num_syst"];
                f.num_samples = file["num_samples"];
                f.cuts        = file["cuts"].get<std::vector<int>>();

                // temp_json is a map between sample numbers and selection branches in the Highland ROOT file (as specified in the .json config file):
                std::map<std::string, std::vector<int>> temp_json = file["samples"];

                // Loop over all the samples of the current file:
                for(const auto& kv : temp_json)
                {
                    // Number of the current sample:
                    const int sam = std::stoi(kv.first);

                    // If the number of the current sample is less than or equal to the total number of samples, we add an entry to the samples map of the FileOptions struct f (this is a map between sample number and a vector containing the selection branch numbers of the Highland ROOT file):
                    if(sam <= num_cov_samples)
                        f.samples.emplace(std::make_pair(sam, kv.second));

                    // If the nuber of the current sample is greater than the total number of samples, an error is thrown:
                    else
                    {
                        std::cout << ERR << "Invalid sample number: " << sam << std::endl;
                        return 64;
                    }
                }

                // Add the file options of the current file to the vector v_files:
                v_files.emplace_back(f);

                // As usable_toys was set to zero previously, this condition will be fulfilled and usable_toys will be set to the number of usable toys as specified in the .json config file:
                if(f.num_toys < usable_toys || usable_toys == 0)
                    usable_toys = f.num_toys;
            }
        }
    }

    // Print some information about which options were set in the .json config file:
    std::cout << TAG << "Output ROOT file: " << fname_output << std::endl
              << TAG << "Toy Weight Cut: " << weight_cut << std::endl
              << TAG << "Calculating Covariance: " << std::boolalpha << do_covariance << std::endl;

    // Print the covariance variables as specified in the .json config file (which are used for the binning):
    std::cout << TAG << "Covariance Variables: ";
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
    std::cout << TAG << "Initalizing histograms." << std::endl;
    std::cout << TAG << "Using " << usable_toys << " toys." << std::endl;

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
    std::cout << TAG << "Finished initializing histograms" << std::endl
              << TAG << "Reading events from files..." << std::endl;

    // Loop over all files specified in the .json config file:
    for(const auto& file : v_files)
    {
        // Declare some variables that will hold the values that we need from the input ROOT files:
        int NTOYS = 0;
        int accum_level[file.num_toys][file.num_samples];
        float hist_variables[nvars][file.num_toys];
        float weight_syst_total_noflux[file.num_toys];
        float weight_syst[file.num_toys][file.num_syst];

        int accum_level_mc[file.num_toys][file.num_samples];
        float hist_variables_mc[nvars];
        float weight_syst_total_noflux_mc;

        // Print some information about which file is opened, which tree is read in, the number of toys and systematics, and which selection branches are mapped to which selection samples:
        std::cout << TAG << "Opening file: " << file.fname_input << std::endl
                  << TAG << "Reading tree: " << file.tree_name << std::endl
                  << TAG << "Num Toys: " << file.num_toys << std::endl
                  << TAG << "Num Syst: " << file.num_syst << std::endl;

        std::cout << TAG << "Branch to Sample mapping:" << std::endl;
        for(const auto& kv : file.samples)
        {
            std::cout << TAG << "Sample " << kv.first << ": ";
            for(const auto& b : kv.second)
                std::cout << b << " ";
            std::cout << std::endl;
        }

        // Open current ROOT input file and access the default tree and the one specified in the .json config file (e.g, all_syst):
        TFile* file_input = TFile::Open(file.fname_input.c_str(), "READ");
        TTree* tree_event = (TTree*)file_input->Get(file.tree_name.c_str());
        TTree* tree_default = (TTree*)file_input->Get("default");

        // Set the branch addresses for the selected tree to the previously declared variables:
        tree_event->SetBranchAddress("NTOYS", &NTOYS);
        tree_event->SetBranchAddress("accum_level", accum_level);
        tree_event->SetBranchAddress("weight_syst", weight_syst);
        tree_event->SetBranchAddress("weight_syst_total", weight_syst_total_noflux);

        // Loop over all the variables specified in the .json config file (e.g., selmu_mom and selmu_costheta):
        for(unsigned int i = 0; i < nvars; ++i)
        {
            // Set the branch address of the current variable of the hist_variables (declared above) array to the variable name:
            tree_event->SetBranchAddress(var_names[i].c_str(), hist_variables[i]);
        }

        // Set the branch addresses for the default tree to the previously declared variables:
        tree_default->SetBranchAddress("accum_level", accum_level_mc);
        tree_default->SetBranchAddress("weight_syst_total", &weight_syst_total_noflux_mc);

        // Loop over all the variables specified in the .json config file (e.g., selmu_mom and selmu_costheta):
        for(unsigned int i = 0; i < nvars; ++i)
        {
            // Set the branch address of the current variable of the hist_variables_mc (declared above) array to the variable name:
            tree_default->SetBranchAddress(var_names[i].c_str(), &hist_variables_mc[i]);
        }

        // Numbers which will hold the number of rejected events, number of events which passed all the cuts, and total number of events:
        unsigned int rejected_weights = 0;
        unsigned int total_weights    = 0;
        unsigned int num_events = tree_event->GetEntries();

        // Print out total number of events:
        std::cout << TAG << "Number of events: " << num_events << std::endl;

        // Loop over all events:
        for(unsigned int i = 0; i < num_events; ++i)
        {
            // Get the ith event:
            tree_event->GetEntry(i);

            // If the number of toys from the input ROOT file does not match the number of toys specified in the .json config file, an error message is printed:
            if(NTOYS != file.num_toys)
                std::cout << ERR << "Incorrect number of toys specified!" << std::endl;

            // Update progress bar every 2000 events (or if it is the last event):
            if(i % 2000 == 0 || i == (num_events - 1))
                pbar.Print(i, num_events - 1);

            // Loop over all toys:
            for(unsigned int t = 0; t < usable_toys; ++t)
            {
                // Loop over all selection samples:
                for(const auto& kv : file.samples)
                {
                    // s is the sample number:
                    unsigned int s = kv.first;

                    // Loop over all selection branches in current selection sample:
                    for(const auto& branch : kv.second)
                    {
                        // Only consider events that passed the selection (accum_level is higher than the given cut for this branch):
                        if(accum_level[t][branch] > file.cuts[branch])
                        {
                            // idx is the bin number for the current event (will be updated below) in the ROOT histograms which have been initialized:
                            int idx = -1;

                            // If do_projection is true, plots are saved with axis as kinematic variable (specified in .json config file, e.g., selmu_mom) instead of bin number. In this case idx will be the value of the current plot variable (e.g., selmu_mom) for the current toy of this event:
                            if(do_projection)
                                idx = hist_variables[var_plot][t];

                            // If do_projection is false, plots are saved with axis as bin number (instead of kinematic variable). In this case idx will be the index of the bin that this event/toy falls in:
                            else
                            {
                                // vars will hold the values of the kinematic variables for the current event and toy:
                                std::vector<double> vars;

                                // Loop over the kinematic variable (e.g., selmu_costheta and selmu_mom) and add the value for the current event and toy to the vars vector:
                                for(unsigned int v = 0; v < nvars; ++v)
                                    vars.push_back(hist_variables[v][t]);
                                
                                // We get the index of the bin for this event and this sample (as specified in the cov_sample_binning files) and set idx equal to it:
                                idx = cov_bin_manager[s].GetBinIndex(vars);
                            }

                            // If "single_syst" has been set to true in the .json config file, the weight for the current event and toy will be the weight given by this single systematic (which needs to be given the correct index in the .json config file). Otherwise the weight will be the total weight for this event and toy:
                            float weight = do_single_syst ? weight_syst[t][syst_idx]
                                                          : weight_syst_total_noflux[t];
                            
                            // If the weight for this event and toy is between zero and the weight cut set in the .json config file, we fill the ROOT histogram for the current sample and toy with the weight of the event in the bin that this event falls in. We also fill the average histogram for the current sample (for all toys) with the weight divided by the number of toys:
                            if(weight > 0.0 && weight < weight_cut)
                            {
                                v_hists[s][t].Fill(idx, weight);
                                v_avg[s].Fill(idx, weight / file.num_toys);
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

        // We now move on to reading out the default tree:
        std::cout << TAG << "Reading default events..." << std::endl;

        // Get the number of events:
        num_events = tree_default->GetEntries();

        // Loop over all events:
        for(unsigned int i = 0; i < num_events; ++i)
        {
            // Get the ith event from the default tree:
            tree_default->GetEntry(i);

            // Loop over all selection samples:
            for(const auto& kv : file.samples)
            {
                // s is the sample number:
                unsigned int s = kv.first;

                // Loop over all selection branches in current selection sample:
                for(const auto& branch : kv.second)
                {
                    // Only consider events that passed the selection (accum_level is higher than the given cut for this branch):
                    if(accum_level_mc[0][branch] > file.cuts[branch])
                    {
                        // idx is the bin number for the current event (will be updated below) in the ROOT histograms which have been initialized:
                        int idx = -1;

                        // If do_projection is true, plots are saved with axis as kinematic variable (specified in .json config file, e.g., selmu_mom) instead of bin number. In this case idx will be the value of the current plot variable (e.g., selmu_mom) for the current event:
                        if(do_projection)
                            idx = hist_variables_mc[var_plot];

                        // If do_projection is false, plots are saved with axis as bin number (instead of kinematic variable). In this case idx will be the index of the bin that this event falls in:
                        else
                        {
                            // vars will hold the values of the kinematic variables for the current event:
                            std::vector<double> vars;

                            // Loop over the kinematic variable (e.g., selmu_costheta and selmu_mom) and add the value for the current event to the vars vector:
                            for(unsigned int v = 0; v < nvars; ++v)
                                vars.push_back(hist_variables_mc[v]);

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

        // Print some information about how many events passed all the selection cuts (total_weights), how many had a weight less than zero or higher than the weight cut (rejected_weights), and their fraction:
        double reject_fraction = (rejected_weights * 1.0) / total_weights;
        std::cout << TAG << "Finished processing events." << std::endl;
        std::cout << TAG << "Total weights: " << total_weights << std::endl;
        std::cout << TAG << "Rejected weights: " << rejected_weights << std::endl;
        std::cout << TAG << "Rejected fraction: " << reject_fraction << std::endl;

        // Close the current ROOT input file and move on to the next one (it this was not the last one):
        file_input->Close();
    }

    // Size of coavariance and correlation matrices (will be updated below):
    unsigned int num_elements = 0;

    // Initialize the covariance and correlation matrices (symmetric ROOT matrices):
    TMatrixTSym<double> cov_mat(num_elements);
    TMatrixTSym<double> cor_mat(num_elements);

    // Vector which will be filled with the MC statistical errors:
    std::vector<float> v_mc_error;

    // If the covariance key is set to true in the .json config file, we compute the covariance and correlation matrices below:
    if(do_covariance)
    {
        std::cout << TAG << "Calculating covariance matrix." << std::endl;

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
            float* w  = v_mc_stat[s].GetArray();

            // Array of the sum of squares of the bin content for current sample s (v_mc_stat was filled from the default tree):
            double* w2 = v_mc_stat[s].GetSumw2()->GetArray();
            //std::cout << "Sample " << s << std::endl;

            // Loop over all bins in current sample:
            for(unsigned int b = 0; b < nbins; ++b)
            {
                //std::cout << "Bin : " << w[b+1] << std::endl;
                //std::cout << "W2  : " << w2[b+1] << std::endl;

                // Compute the relative error for current sample s and current bin b:
                float rel_error = w2[b+1] / (w[b+1] * w[b+1]);

                // Add the relative error for current sample and bin to the v_mc_error vector:
                v_mc_error.emplace_back(rel_error);
            }
        }

        // Print information about the number of toys we are using:
        std::cout << TAG << "Using " << usable_toys << " toys." << std::endl;

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
                for(unsigned int j = 0; j < num_elements; ++j)
                {
                    // Only compute the matrix element, if the denominators in the expression below are both not zero:
                    if(v_mean[i] != 0 && v_mean[j] != 0)
                    {
                        // Compute current matrix element of the covariance matrix:
                        cov_mat(i, j) += (1.0 - v_toys[t][i] / v_mean[i])
                                         * (1.0 - v_toys[t][j] / v_mean[j]) / (1.0 * usable_toys);
                    }
                }
            }
        }

        // If the "mc_stat_error" key was set to true in the .json config file, we add the MC statistical error to the diagonal entries of the covariance matrix:
        if(do_mc_stat)
        {
            std::cout << TAG << "Adding MC stat error to covariance." << std::endl;

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
            for(unsigned int j = 0; j < num_elements; ++j)
            {
                // Get the diagonal entries of the covariance matrix:
                double bin_i  = cov_mat(i, i);
                double bin_j  = cov_mat(j, j);

                // Compute current matrix element of the correlation matrix:
                cor_mat(i, j) = cov_mat(i, j) / std::sqrt(bin_i * bin_j);

                // If the current matrix element is a nan, we set it to zero:
                if(std::isnan(cor_mat(i, j)))
                    cor_mat(i, j) = 0;
            }
        }
    }

    std::cout << TAG << "Saving to output file." << std::endl;

    // Create the output file. If it already exists, it will be overwritten:
    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output->cd();

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

    std::cout << TAG << "Finished." << std::endl;

    // Print Arigatou Gozaimashita with Rainbowtext :)
    std::cout << TAG << color::RainbowText("\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01")
              << std::endl;
    
    return 0;
}
