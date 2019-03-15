#include "FitObj.hh"

FitObj::FitObj(const std::string& json_config, const std::string& event_tree_name,
               bool is_true_tree)
{
    OptParser parser;
    if(!parser.ParseJSON(json_config))
    {
        std::cerr << ERR << "JSON parsing failed. Exiting.\n";
        return;
    }
    parser.PrintOptions();
    m_tree_type = is_true_tree ? "true" : "selected";
    m_fit_par   = nullptr;
    m_flux_par  = nullptr;

    std::string input_dir = parser.input_dir;
    std::string fname_data = parser.fname_data;
    std::string fname_mc = parser.fname_mc;
    std::string fname_output = parser.fname_output;
    std::vector<std::string> topology = parser.sample_topology;

    npar = 0;
    const double potD = parser.data_POT;
    const double potMC = parser.mc_POT;
    m_norm = potD / potMC;
    m_threads = parser.num_threads;
    signal_def = parser.signal_definition;

    std::cout << TAG << "Configure file parsing finished." << std::endl;
    std::cout << TAG << "Opening " << fname_data << " for data selection.\n"
              << TAG << "Opening " << fname_mc << " for MC selection." << std::endl;

    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    for(const auto& opt : parser.samples)
    {
        if(opt.use_sample == true)
        {
            std::cout << TAG << "Adding new sample to fit.\n"
                      << TAG << "Name: " << opt.name << std::endl
                      << TAG << "CutB: " << opt.cut_branch << std::endl
                      << TAG << "Detector: " << opt.detector << std::endl
                      << TAG << "True tree: " << std::boolalpha << is_true_tree << std::endl
                      << TAG << "Use Sample: " << std::boolalpha << opt.use_sample << std::endl;

            auto s = new AnaSample(opt.cut_branch, opt.name, opt.detector, opt.binning, tdata);
            s->SetNorm(potD / potMC);
            if(opt.cut_branch >= 0 && !is_true_tree)
                samples.push_back(s);
            else if(opt.cut_branch < 0 && is_true_tree)
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
    enubins.push_back(nd_numu_bins->GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins->GetNbins(); ++i)
        enubins.push_back(nd_numu_bins->GetBinUpEdge(i + 1));

    TMatrixDSym* cov_flux = (TMatrixDSym*)finfluxcov -> Get(parser.flux_cov.matrix.c_str());
    finfluxcov->Close();

    std::cout << TAG << "Setup Xsec Covariance" << std::endl;
    TFile* file_xsec_cov = TFile::Open(parser.xsec_cov.fname.c_str(), "READ");
    std::cout << TAG << "Opening " << parser.xsec_cov.fname << " for xsec covariance." << std::endl;
    TMatrixDSym* cov_xsec = (TMatrixDSym*)file_xsec_cov -> Get(parser.xsec_cov.matrix.c_str());
    file_xsec_cov -> Close();

    std::cout << TAG << "Setup Detector Covariance" << std::endl;
    TFile* file_detcov = TFile::Open(parser.det_cov.fname.c_str(), "READ");
    TMatrixDSym* cov_det_in = (TMatrixDSym*)file_detcov -> Get(parser.det_cov.matrix.c_str());
    TMatrixDSym cov_det = *cov_det_in;
    file_detcov -> Close();

    // FitParameters sigfitpara("par_fit", false);
    FitParameters* sigfitpara = new FitParameters("par_fit", false);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            sigfitpara->AddDetector(opt.name, parser.signal_definition);
    }
    sigfitpara->InitEventMap(samples, 0);
    fit_par.push_back(sigfitpara);
    m_fit_par = sigfitpara;
    npar += sigfitpara->GetNpar();

    // FluxParameters fluxpara("par_flux");
    FluxParameters* fluxpara = new FluxParameters("par_flux");
    if(parser.flux_cov.do_fit)
    {
        fluxpara->SetCovarianceMatrix(*cov_flux, parser.flux_cov.decompose);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                fluxpara->AddDetector(opt.name, enubins);
        }
        fluxpara->InitEventMap(samples, 0);
        fit_par.push_back(fluxpara);
        m_flux_par = fluxpara;
        npar += fluxpara->GetNpar();
    }

    // XsecParameters xsecpara("par_xsec");
    XsecParameters* xsecpara = new XsecParameters("par_xsec");
    if(parser.xsec_cov.do_fit)
    {
        xsecpara->SetCovarianceMatrix(*cov_xsec, parser.xsec_cov.decompose);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                xsecpara->AddDetector(opt.name, opt.xsec);
        }
        xsecpara->InitEventMap(samples, 0);
        fit_par.push_back(xsecpara);
        npar += xsecpara->GetNpar();
    }

    // DetParameters detpara("par_det");
    DetParameters* detpara = new DetParameters("par_det");
    if(parser.det_cov.do_fit)
    {
        detpara->SetCovarianceMatrix(cov_det, parser.det_cov.decompose);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                detpara->AddDetector(opt.name, samples, true);
        }
        if(is_true_tree)
            detpara->InitEventMap(samples, 2);
        else
            detpara->InitEventMap(samples, 0);
        fit_par.push_back(detpara);
        npar += detpara->GetNpar();
    }

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

        std::cout << TAG << "Adding signal " << sig.name << " with ID " << signal_id << " to fit."
                  << std::endl;

        signal_bins.emplace_back(BinManager(sig.binning));
        signal_id++;
    }

    total_signal_bins = 0;
    for(int i = 0; i < signal_id; ++i)
    {
        std::stringstream ss;
        ss << "hist_" << m_tree_type << "_signal_" << i;

        const int nbins = signal_bins.at(i).GetNbins();
        signal_hist.emplace_back(TH1D(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));
        total_signal_bins += nbins;
    }
}

void FitObj::ReweightEvents(const std::vector<double>& input_par)
{
    ResetHist();

    std::vector<std::vector<double>> new_par;
    auto start = input_par.begin();
    auto end = input_par.begin();
    for(const auto& param_type : fit_par)
    {
        start = end;
        end = start + param_type->GetNpar();
        new_par.emplace_back(std::vector<double>(start, end));
    }

    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s]->GetN();
        const std::string det(samples[s]->GetDetector());
#pragma omp parallel for num_threads(m_threads)
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s]->GetEvent(i);
            ev->ResetEvWght();
            for(int f = 0; f < fit_par.size(); ++f)
                fit_par[f]->ReWeight(ev, det, s, i, new_par.at(f));
        }
    }

    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s]->GetN();
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s]->GetEvent(i);
            if(ev->isSignalEvent())
            {
                int signal_id = ev->GetSignalType();
                int bin_idx = signal_bins[signal_id].GetBinIndex(
                    std::vector<double>{ev->GetTrueD2(), ev->GetTrueD1()});
                signal_hist[signal_id].Fill(bin_idx + 0.5, ev->GetEvWght());
            }
        }
    }

    for(auto& hist : signal_hist)
        hist.Scale(m_norm);
}

void FitObj::ReweightNominal()
{
    ResetHist();

    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s]->GetN();
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s]->GetEvent(i);
            if(ev->isSignalEvent())
            {
                int signal_id = ev->GetSignalType();
                int bin_idx = signal_bins[signal_id].GetBinIndex(
                    std::vector<double>{ev->GetTrueD2(), ev->GetTrueD1()});
                signal_hist[signal_id].Fill(bin_idx + 0.5, ev->GetEvWghtMC());
            }
        }
    }

    for(auto& hist : signal_hist)
        hist.Scale(m_norm);
}

void FitObj::ResetHist()
{
    for(auto& hist : signal_hist)
        hist.Reset();
}

TH1D FitObj::GetHistCombined(const std::string& suffix) const
{
    std::string hist_name = m_tree_type + "_signal_" + suffix;
    TH1D hist_combined(hist_name.c_str(), hist_name.c_str(), total_signal_bins, 0,
                       total_signal_bins);

    unsigned int bin = 1;
    for(const auto& hist : signal_hist)
    {
        for(int i = 1; i <= hist.GetNbinsX(); ++i)
            hist_combined.SetBinContent(bin++, hist.GetBinContent(i));
    }

    return hist_combined;
}

double FitObj::ReweightFluxHist(const std::vector<double>& input_par, TH1D& flux_hist,
                                const std::string& det)
{
    double flux_int = 0.0;
    if(m_flux_par == nullptr)
        flux_int = flux_hist.Integral();

    else
    {
        std::vector<double> flux_par;
        auto start = input_par.begin();
        auto end = input_par.begin();
        for(const auto& param_type : fit_par)
        {
            start = end;
            end = start + param_type->GetNpar();
            if(param_type == m_flux_par)
                flux_par.assign(start, end);
        }

        for(int i = 1; i <= flux_hist.GetNbinsX(); ++i)
        {
            const double enu = flux_hist.GetBinCenter(i);
            const double val = flux_hist.GetBinContent(i);
            const int idx = m_flux_par->GetBinIndex(det, enu)
                            + m_flux_par->GetDetectorOffset(det);
            flux_int += val * flux_par[idx];
        }
    }

    return flux_int;
}

TMatrixDSym FitObj::GetPrefitCov()
{
    TMatrixDSym prefit_cov(npar);
    TMatrixDSym temp_mat = CalcTemplateCov();

    unsigned int idx = 0;
    for(const auto& param_type : fit_par)
    {
        TMatrixDSym* cov_mat = nullptr;
        if(param_type == m_fit_par)
            cov_mat = &temp_mat;
        else
            cov_mat = param_type->GetOriginalCovMat();

        for(int i = 0; i < cov_mat->GetNrows(); ++i)
        {
            for(int j = 0; j < cov_mat->GetNrows(); ++j)
            {
                prefit_cov(idx+i, idx+j) = (*cov_mat)(i,j);
            }
        }

        idx += cov_mat->GetNrows();
    }

    return prefit_cov;
}

TMatrixDSym FitObj::CalcTemplateCov()
{
    TMatrixDSym temp_cov(m_fit_par->GetNpar());

    ReweightNominal();
    auto signal_hist = GetHistCombined("prefit");

    for(int i = 0; i < m_fit_par->GetNpar(); ++i)
        temp_cov(i,i) = 1.0 / signal_hist.GetBinContent(i+1);

    return temp_cov;
}
