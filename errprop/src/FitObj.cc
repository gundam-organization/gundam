#include "FitObj.hh"

FitObj::FitObj(const std::string& json_config, const std::string& event_tree_name, bool is_true_tree)
{
    OptParser parser;
    if(!parser.ParseJSON(json_config))
    {
        std::cerr << ERR << "JSON parsing failed. Exiting.\n";
        return;
    }
    parser.PrintOptions();

    std::string input_dir = parser.input_dir;
    std::string fname_data = parser.fname_data;
    std::string fname_mc   = parser.fname_mc;
    std::string fname_output = parser.fname_output;
    std::vector<std::string> topology = parser.sample_topology;

    const double potD  = parser.data_POT;
    const double potMC = parser.mc_POT;
    m_threads = parser.num_threads;

    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    std::cout << TAG << "Configure file parsing finished." << std::endl;
    std::cout << TAG << "Opening " << fname_data << " for data selection.\n"
              << TAG << "Opening " << fname_mc << " for MC selection." << std::endl;

    for(const auto& opt : parser.samples)
    {
        if(opt.use_sample == true)
        {
            std::cout << TAG << "Adding new sample to fit.\n"
                      << TAG << "Name: " << opt.name << std::endl
                      << TAG << "CutB: " << opt.cut_branch << std::endl
                      << TAG << "Detector: " << opt.detector << std::endl
                      << TAG << "Use Sample: " << std::boolalpha << opt.use_sample << std::endl;

            auto s = new AnaSample(opt.cut_branch, opt.name, opt.detector, opt.binning, tdata);
            s -> SetNorm(potD/potMC);
            if(opt.cut_branch >= 0)
                samples.push_back(s);
        }
    }

    AnaTreeMC event_tree(fname_mc.c_str(), event_tree_name);
    std::cout << TAG << "Reading and collecting events." << std::endl;
    std::cout << TAG << "Tree: " << event_tree_name << std::endl;
    event_tree.GetEvents(samples, parser.signal_definition, is_true_tree);

    TFile* finfluxcov = TFile::Open(parser.flux_cov.fname.c_str(), "READ");
    std::cout << TAG << "Opening " << parser.flux_cov.fname << " for flux covariance." << std::endl;
    TH1D* nd_numu_bins_hist = (TH1D*)finfluxcov->Get(parser.flux_cov.binning.c_str());
    TAxis* nd_numu_bins = nd_numu_bins_hist->GetXaxis();

    std::vector<double> enubins;
    enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));
    finfluxcov -> Close();

    //FitParameters sigfitpara("par_fit", false);
    FitParameters* sigfitpara = new FitParameters("par_fit", false);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            sigfitpara->AddDetector(opt.name, parser.signal_definition);
    }
    sigfitpara->InitEventMap(samples, 0);
    fit_par.push_back(sigfitpara);

    //FluxParameters fluxpara("par_flux");
    FluxParameters* fluxpara = new FluxParameters("par_flux");
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            fluxpara->AddDetector(opt.name, enubins);
    }
    fluxpara->InitEventMap(samples, 0);
    fit_par.push_back(fluxpara);

    //XsecParameters xsecpara("par_xsec");
    XsecParameters* xsecpara = new XsecParameters("par_xsec");
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            xsecpara->AddDetector(opt.name, opt.xsec);
    }
    xsecpara->InitEventMap(samples, 0);
    fit_par.push_back(xsecpara);

    //DetParameters detpara("par_det");
    DetParameters* detpara = new DetParameters("par_det");
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            detpara->AddDetector(opt.name, samples, true);
    }
    detpara->InitEventMap(samples, 0);
    fit_par.push_back(detpara);

    InitSignalHist(parser.signal_definition);
    std::cout << TAG << "Finished initialization." << std::endl;
}

FitObj::~FitObj()
{
    for(unsigned int i = 0; i < fit_par.size(); ++i)
        delete fit_par.at(i);

    for(unsigned int i = 0; i < samples.size(); ++i)
        delete samples.at(i);
}

void FitObj::InitSignalHist(const std::vector<SignalDef>& v_signal)
{
    unsigned int signal_id = 0;
    for(const auto& sig : v_signal)
    {
        if(sig.use_signal == false)
            continue;

        std::cout << TAG << "Adding signal " << sig.name << " with ID " << signal_id
                  << " to fit." << std::endl;

        signal_bins.emplace_back(BinManager(sig.binning));
        signal_id++;
    }

    for(int i = 0; i < signal_id; ++i)
    {
        std::stringstream ss;
        ss << "hist_signal_" << i;

        const int nbins = signal_bins.at(i).GetNbins();
        signal_hist.emplace_back(TH1D(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));
        signal_hist[i].SetDirectory(0);
    }
}

void FitObj::ReweightEvents(const std::vector<double>& input_par)
{
    std::vector<std::vector<double>> new_par;
    auto start = input_par.begin();
    auto end = input_par.begin();
    for(const auto& param_type : fit_par)
    {
        start = end;
        end = start + param_type -> GetNpar();
        new_par.emplace_back(std::vector<double>(start, end));
    }

    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s] -> GetN();
        const std::string det(samples[s] -> GetDetector());
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s] -> GetEvent(i);
            ev -> ResetEvWght();
            for(int f = 0; f < fit_par.size(); ++f)
                fit_par[f] -> ReWeight(ev, det, s, i, new_par.at(f));

            if(ev -> isSignalEvent())
            {
                int signal_id = ev -> GetSignalType();
                int bin_idx = signal_bins[signal_id].GetBinIndex(std::vector<double>{ev->GetTrueD1(),ev->GetTrueD2()});
                signal_hist[signal_id].Fill(bin_idx + 0.5, ev->GetEvWght());
            }
        }
    }
}
