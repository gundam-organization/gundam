#include "AnaSample.hh"
#include "Logger.h"
#include "TTreeFormula.h"

using xsllh::FitBin;

// ctor
AnaSample::AnaSample(int sample_id, const std::string& name, const std::string& detector,
                     const std::string& binning, TTree* t_data)
    : m_sample_id(sample_id)
    , m_name(name)
    , m_detector(detector)
    , m_binning(binning)
    , m_data_tree(t_data)
    , m_norm(1.0)
{
    Reset();
}

AnaSample::AnaSample(const SampleOpt& sample, TTree* t_data){

    m_sample_id = sample.cut_branch;
    m_name = sample.name;
    m_detector = sample.detector;
    m_binning = sample.binning;
    m_additional_cuts = sample.additional_cuts;
    m_data_POT = sample.data_POT;
    m_mc_POT = sample.mc_POT;
    m_norm = 1.0;

    m_data_tree = t_data;

    this->Reset();

}

// Private constructor
void AnaSample::Reset() {

    Logger::setUserHeaderStr("[AnaSample]");
    m_additional_cuts_formulae = new TTreeFormula(
        "additional_cuts", m_additional_cuts.c_str(), m_data_tree
    );

    TH1::SetDefaultSumw2(true);
    SetBinning(m_binning);

    if(m_data_POT != 0 and m_mc_POT != 0){
        SetNorm(m_data_POT/m_mc_POT);
    }

    LogInfo << m_name << ", ID " << m_sample_id << std::endl
             << "Detector: " << m_detector << std::endl
             << "Bin edges: " << std::endl;

//    for(const auto& bin : m_bin_edges)
//    {
//        std::cout << bin.D2low << " " << bin.D2high << " " << bin.D1low << " " << bin.D1high
//                  << std::endl;
//    }

    m_hpred    = nullptr;
    m_hmc      = nullptr;
    m_hmc_true = nullptr;
    m_hsig     = nullptr;
    m_hdata    = nullptr;

    m_llh = new PoissonLLH;

    MakeHistos(); // with default binning

    LogInfo << "MakeHistos called." << std::endl;

}

AnaSample::~AnaSample()
{
    if(m_hpred != nullptr)
        delete m_hpred;
    if(m_hmc != nullptr)
        delete m_hmc;
    if(m_hmc_true != nullptr)
        delete m_hmc_true;
    if(m_hdata != nullptr)
        delete m_hdata;
    if(m_hsig != nullptr)
        delete m_hsig;
}

void AnaSample::SetBinning(const std::string& binning)
{
    m_binning = binning;
    m_nbins   = 0;

    std::ifstream fin(m_binning, std::ios::in);
    if(!fin.is_open())
    {
        LogError << "In AnaSample::SetBinning().\n"
                 << "Failed to open binning file: " << m_binning << std::endl;
    }
    else
    {
        std::string line;
        while(std::getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss >> D2_1 >> D2_2 >> D1_1 >> D1_2))
            {
                LogError << "Bad line format: " << line << std::endl;
                continue;
            }
            m_bin_edges.emplace_back(FitBin(D1_1, D1_2, D2_1, D2_2));
        }
        m_nbins = m_bin_edges.size();
    }
}

// Mapping the Highland topology codes to consecutive integers:
void AnaSample::SetTopologyHLCode(const std::vector<int>& HLTopologyCodes)
{
    for(std::size_t i=0; i < HLTopologyCodes.size(); ++i)
    {
        topology_HL_code[i] = HLTopologyCodes[i];
    }
}

// ClearEvents -- clears all events from event vector
void AnaSample::ClearEvents() { m_events.clear(); }

// GetN -- get number of events stored
int AnaSample::GetN() const { return m_events.size(); }

AnaEvent* AnaSample::GetEvent(int evnum)
{
    if(m_events.empty())
    {
        LogError << "In AnaSample::GetEvent()" << std::endl;
        LogError << "No events are found in " << m_name << " sample." << std::endl;
        return nullptr;
    }
    else if(evnum >= m_events.size())
    {
        LogError << "In AnaSample::GetEvent()" << std::endl;
        LogError << "Event number out of bounds in " << m_name << " sample." << std::endl;
        return nullptr;
    }

    return &m_events.at(evnum);
}

std::vector<AnaEvent>& AnaSample::GetEventList(){
    return m_events;
}

void AnaSample::AddEvent(AnaEvent& event) {
    m_events.emplace_back(event);
    // since default constructor doesn't do it by itself
    m_events.back().HookIntMembers();
    m_events.back().HookFloatMembers();
}

void AnaSample::ResetWeights()
{
    for(auto& event : m_events)
        event.SetEvWght(1.0);
}

void AnaSample::PrintStats() const
{
    double mem_kb = sizeof(m_events) * m_events.size() / 1000.0;
    LogInfo << "Sample " << m_name << " ID = " << m_sample_id << std::endl;
    LogInfo << "Num of events = " << m_events.size() << std::endl;
    LogInfo << "Memory used   = " << mem_kb << " kB." << std::endl;
}

void AnaSample::MakeHistos()
{
    if(m_hpred != nullptr)
        delete m_hpred;
    m_hpred = new TH1D(Form("%s_pred_recD1D2", m_name.c_str()),
                       Form("%s_pred_recD1D2", m_name.c_str()), m_nbins, 0, m_nbins);
    m_hpred->SetDirectory(0);

    if(m_hmc != nullptr)
        delete m_hmc;
    m_hmc = new TH1D(Form("%s_mc_recD1D2", m_name.c_str()), Form("%s_mc_recD1D2", m_name.c_str()),
                     m_nbins, 0, m_nbins);
    m_hmc->SetDirectory(0);

    if(m_hmc_true != nullptr)
        delete m_hmc_true;
    m_hmc_true = new TH1D(Form("%s_mc_trueD1D2", m_name.c_str()),
                          Form("%s_mc_trueD1D2", m_name.c_str()), m_nbins, 0, m_nbins);
    m_hmc_true->SetDirectory(0);

    if(m_hsig != nullptr)
        delete m_hsig;
    m_hsig = new TH1D(Form("%s_mc_trueSignal", m_name.c_str()),
                      Form("%s_mc_trueSignal", m_name.c_str()), m_nbins, 0, m_nbins);
    m_hsig->SetDirectory(0);

    LogInfo << m_nbins << " bins inside MakeHistos()." << std::endl;
}

void AnaSample::SetData(TObject* hdata)
{
    if(m_hdata != nullptr)
        delete m_hdata;
    m_hdata = (TH1D*)hdata->Clone(Form("%s_data", m_name.c_str()));
    m_hdata->SetDirectory(0);
}

int AnaSample::GetBinIndex(const double D1, const double D2) const
{
    for(int i = 0; i < m_bin_edges.size(); ++i)
    {
        if(D1 >= m_bin_edges[i].D1low && D1 < m_bin_edges[i].D1high && D2 >= m_bin_edges[i].D2low
           && D2 < m_bin_edges[i].D2high)
        {
            return i;
        }
    }
    return -1;
}

void AnaSample::FillEventHist(int datatype, bool stat_fluc){

    if(m_hpred != nullptr) m_hpred->Reset();
    if(m_hmc != nullptr) m_hmc->Reset();
    if(m_hmc_true != nullptr) m_hmc_true->Reset();
    if(m_hsig != nullptr) m_hsig->Reset();

    for(std::size_t iEvent = 0; iEvent < m_events.size(); ++iEvent){

        double D1_rec  = m_events[iEvent].GetRecoD1();
        double D2_rec  = m_events[iEvent].GetRecoD2();
        double D1_true = m_events[iEvent].GetTrueD1();
        double D2_true = m_events[iEvent].GetTrueD2();
        double wght    = datatype >= 0 ? m_events[iEvent].GetEvWght() : m_events[iEvent].GetEvWghtMC();

        int anybin_index_rec  = GetBinIndex(D1_rec, D2_rec);
        int anybin_index_true = GetBinIndex(D1_true, D2_true);

        m_hpred->Fill(anybin_index_rec + 0.5, wght);
        m_hmc->Fill(anybin_index_rec + 0.5, wght);
        m_hmc_true->Fill(anybin_index_true + 0.5, wght);

        if(m_events[iEvent].isSignalEvent())
            m_hsig->Fill(anybin_index_true + 0.5, wght);
    }

    m_hpred->Scale(m_norm);
    m_hmc->Scale(m_norm);
    m_hsig->Scale(m_norm);

    if(datatype == 0 || datatype == -1)
        return;

    else if(datatype == 1)
    {
        SetData(m_hpred);
        m_hdata->Reset();

        if(stat_fluc)
            LogInfo << "Applying statistical fluctuations..." << std::endl;

        for(int j = 1; j <= m_hpred->GetNbinsX(); ++j)
        {
            double val = m_hpred->GetBinContent(j);
            if(stat_fluc)
                val = gRandom->Poisson(val);
#ifndef NDEBUG
            if(val == 0.0)
            {
                LogWarning << "In AnaSample::FillEventHist()\n"
                           << m_name << " bin " << j
                           << " has 0 entries. This may cause a problem with chi2 computations."
                           << std::endl;
                continue;
            }
#endif
            m_hdata->SetBinContent(j, val);
        }
    }

    else if(datatype == 2 || datatype == 3)
    {
        SetData(m_hpred);
        m_hdata->Reset();

        float D1_rec_tree, D2_rec_tree, wght;
        int cut_branch;

        m_data_tree->SetBranchAddress("cut_branch", &cut_branch);
        m_data_tree->SetBranchAddress("weight", &wght);
        m_data_tree->SetBranchAddress("D1Reco", &D1_rec_tree);
        m_data_tree->SetBranchAddress("D2Reco", &D2_rec_tree);

        m_additional_cuts_formulae->SetTree(m_data_tree);
        m_data_tree->SetNotify(m_additional_cuts_formulae); // This is needed only for TChain.

        long int n_entries = m_data_tree->GetEntries();
        for(std::size_t i = 0; i < n_entries; ++i)
        {
            m_data_tree->GetEntry(i);
            if(cut_branch != m_sample_id)
                continue;

            bool doesEventPassAdditionalCuts = false;
            for(int jInstance = 0; jInstance < m_additional_cuts_formulae->GetNdata(); jInstance++) {
                if (m_additional_cuts_formulae->EvalInstance(jInstance) ) {
                    doesEventPassAdditionalCuts = true;
                    break;
                }
            }

            if(not doesEventPassAdditionalCuts){
                continue;
            }

            int anybin_index = GetBinIndex(D1_rec_tree, D2_rec_tree);
            if(anybin_index != -1)
            {
                m_hdata->Fill(anybin_index + 0.5, wght);
            }
#ifndef NDEBUG
            else
            {
                LogWarning << "In AnaSample::FillEventHist()\n"
                           << "No bin for current data event.\n"
                           << "D1 Reco: " << D1_rec_tree << std::endl
                           << "D2 Reco: " << D2_rec_tree << std::endl;
            }
#endif
        }

        if(stat_fluc && datatype == 2)
        {
            if(stat_fluc)
                LogInfo << "Applying statistical fluctuations..." << std::endl;

            for(unsigned int i = 1; i <= m_hdata->GetNbinsX(); ++i)
            {
                double val = gRandom->Poisson(m_hdata->GetBinContent(i));
                m_hdata->SetBinContent(i, val);
            }
        }

#ifndef NDEBUG
        LogInfo << "Data histogram filled: " << std::endl;
        m_hdata->Print();
#endif
    }

    else
    {
        LogWarning << "In AnaSample::FillEventHist()\n"
                   << "Invalid data type to fill histograms!\n";
    }

}

void AnaSample::SetLLHFunction(const std::string& func_name)
{
    if(m_llh != nullptr)
        delete m_llh;

    if(func_name.empty())
    {
        LogInfo << "Likelihood function name empty. Setting to Poisson by default." << std::endl;
        m_llh = new PoissonLLH;
    }
    else if(func_name == "Poisson")
    {
        LogInfo << "Setting likelihood function to Poisson." << std::endl;
        m_llh = new PoissonLLH;
    }
    else if(func_name == "Effective")
    {
        LogInfo << "Setting likelihood function to Tianlu's effective likelihood." << std::endl;
        m_llh = new EffLLH;
    }
    else if(func_name == "Barlow")
    {
        LogInfo << "Setting likelihood function to Barlow-Beeston." << std::endl;
        m_llh = new BarlowLLH;
    }
}

// Compute the statistical chi2 contribution from this sample based on the current m_hpred and m_hdata histograms:
double AnaSample::CalcLLH() const
{
    if(m_hdata == nullptr)
    {
        LogError << "In AnaSample::CalcLLH()\n"
                  << "Need to define data histogram." << std::endl;
        throw std::runtime_error("m_hdata is a nullptr.");
    }

    // Number of reco bins as specified in binning file:
    const unsigned int nbins = m_hpred->GetNbinsX();

    // Array of the number of expected events in each bin:
    double* exp_w  = m_hpred->GetArray();

    // Array of sum of squares of weights in each bin for the expected events:
    double* exp_w2 = m_hpred->GetSumw2()->GetArray();

    // Array of the number of measured events in data in each bin:
    double* data   = m_hdata->GetArray();

    // Initialize chi2 variable which will be updated below and then returned:
    double chi2 = 0.0;

    // Loop over all bins:
    for(unsigned int i = 1; i <= nbins; ++i)
        {
            // Compute chi2 contribution from current bin (done in Likelihoods.hh):
            chi2 += (*m_llh)(exp_w[i], exp_w2[i], data[i]);
        }

    // Sum of the chi2 contributions for each bin is returned:
    return chi2;
}

double AnaSample::CalcChi2() const
{
    if(m_hdata == nullptr)
    {
        LogError << "In AnaSample::CalcChi2()\n"
                  << "Need to define data histogram." << std::endl;
        return 0.0;
    }

    int nbins = m_hpred->GetNbinsX();
    if(nbins != m_hdata->GetNbinsX())
    {
        LogError << "In AnaSample::CalcChi2()\n"
                  << "Binning mismatch between data and mc.\n"
                  << "MC bins: " << nbins << ", Data bins: " << m_hdata->GetNbinsX()
                  << std::endl;
        return 0.0;
    }

    double chi2 = 0.0;
    for(int j = 1; j <= nbins; ++j)
    {
        double obs = m_hdata->GetBinContent(j);
        double exp = m_hpred->GetBinContent(j);
        if(exp > 0.0)
        {
            // added when external fake datasets (you cannot reweight when simply 0)
            // this didn't happen when all from same MC since if exp=0 then obs =0

            chi2 += 2 * (exp - obs);
            if(obs > 0.0)
                chi2 += 2 * obs * TMath::Log(obs / exp);

            if(chi2 < 0.0)
            {
#ifndef NDEBUG
                LogWarning << "In AnaSample::CalcChi2()\n"
                          << "Stat chi2 is less than 0: " << chi2 << ", setting to 0."
                          << std::endl;
                LogWarning << "exp and obs is: " << exp << " and " << obs << "."
                          << std::endl;
#endif
                chi2 = 0.0;
            }
        }
    }

    if(chi2 != chi2)
    {
        LogWarning << "In AnaSample::CalcChi2()\n"
                  << "Stat chi2 is nan, setting to 0." << std::endl;
        chi2 = 0.0;
    }

    return chi2;
}

double AnaSample::CalcEffLLH() const
{
    const unsigned int nbins = m_hpred->GetNbinsX();
    double* exp_w  = m_hpred->GetArray();
    double* exp_w2 = m_hpred->GetSumw2()->GetArray();
    double* data   = m_hdata->GetArray();

    //std::cout << m_name << std::endl;

    double llh_eff = 0.0;
    for(unsigned int i = 1; i <= nbins; ++i)
    {
        if(exp_w[i] <= 0.0)
            continue;

        const double b = exp_w[i] / exp_w2[i];
        const double a = (exp_w[i] * b) + 1.0;
        const double k = data[i];

        //std::cout << "--------------" << std::endl;
        //std::cout << "i  : " << i << std::endl;
        //std::cout << "w  : " << exp_w[i] << std::endl;
        //std::cout << "w2 : " << exp_w2[i] << std::endl;
        //std::cout << "a  : " << a << std::endl;
        //std::cout << "b  : " << b << std::endl;
        //std::cout << "k  : " << data[i] << std::endl;

        llh_eff += a * std::log(b) + std::lgamma(k+a) - std::lgamma(k+1) - ((k+a) * std::log1p(b)) - std::lgamma(a);
    }

    return -2 * llh_eff;
}

void AnaSample::Write(TDirectory* dirout, const std::string& bsname, int fititer)
{
    dirout->cd();
    m_hpred->Write(Form("%s_pred", bsname.c_str()));
    m_hmc_true->Write(Form("%s_true", bsname.c_str()));
    if(fititer == 0)
    {
        m_hmc->Write(Form("%s_mc", bsname.c_str()));
        if(m_hdata != nullptr)
            m_hdata->Write(Form("%s_data", bsname.c_str()));
    }
}

void AnaSample::GetSampleBreakdown(TDirectory* dirout, const std::string& tag,
                                   const std::vector<std::string>& topology, bool save)
{
    const int ntopology = topology.size();
    int compos[ntopology];
    std::vector<TH1D> hAnybin_rec;
    std::vector<TH1D> hAnybin_true;

    for(int i = 0; i < ntopology; ++i)
    {
        compos[i] = 0;
        hAnybin_rec.emplace_back(
            TH1D(Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 m_nbins, 0, m_nbins));
        hAnybin_rec[i].SetDirectory(0);
        hAnybin_rec[i].GetXaxis()->SetTitle("Bin Index");

        hAnybin_true.emplace_back(
            TH1D(Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 m_nbins, 0, m_nbins));
        hAnybin_true[i].SetDirectory(0);
        hAnybin_true[i].GetXaxis()->SetTitle("Bin Index");
    }

    int Ntot = GetN();

    // Loop over all events:
    for(std::size_t i = 0; i < m_events.size(); ++i)
    {
        double D1_rec    = m_events[i].GetRecoD1();
        double D2_rec    = m_events[i].GetRecoD2();
        double D1_true   = m_events[i].GetTrueD1();
        double D2_true   = m_events[i].GetTrueD2();
        double wght      = m_events[i].GetEvWght();
        int evt_topology = m_events[i].GetTopology();

        compos[evt_topology]++;
        int anybin_index_rec  = GetBinIndex(D1_rec, D2_rec);
        int anybin_index_true = GetBinIndex(D1_true, D2_true);

        // Fill histogram for this topolgy with the current event:
        hAnybin_rec[topology_HL_code[evt_topology]].Fill(anybin_index_rec + 0.5, wght);
        hAnybin_true[topology_HL_code[evt_topology]].Fill(anybin_index_true + 0.5, wght);
    }

    dirout->cd();
    for(int i = 0; i < ntopology; ++i)
    {
        hAnybin_true[i].Scale(m_norm);
        hAnybin_rec[i].Scale(m_norm);

        if(save == true)
        {
            hAnybin_true[i].Write();
            hAnybin_rec[i].Write();
        }
    }

    LogInfo << "GetSampleBreakdown()\n"
              << "============ Sample " << m_name << " ===========" << std::endl;

    for(int j = 0; j < ntopology; ++j)
    {
        std::cout << std::setw(10) << topology[j] << std::setw(5) << j << std::setw(5) << compos[j]
                  << std::setw(10) << ((1.0 * compos[j]) / Ntot) * 100.0 << "%" << std::endl;
    }

    std::cout << std::setw(10) << "Total" << std::setw(5) << " " << std::setw(5) << Ntot
              << std::setw(10) << "100.00%" << std::endl;
}
