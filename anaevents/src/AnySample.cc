//////////////////////////////////////////////////////////
//
//  A class for event samples for for Any analysis
//
//
//
//  Created: Nov 17 2015
//
//////////////////////////////////////////////////////////
#include "AnySample.hh"

// ctor
AnySample::AnySample(int sample_id, const std::string& name,
        std::vector<std::pair <double,double> > v_d1edges,
        std::vector<std::pair <double,double> > v_d2edges,
        TTree* data, bool isBuffer, bool isEmpty, bool isIngrid)
{
    m_sampleid = sample_id; //unique id
    m_name     = name;      //some comprehensible name
    m_data_tree = data;
    m_D1edges = v_d1edges;
    m_D2edges = v_d2edges;
    m_empty = isEmpty;
    m_BufferBin = isBuffer;
    m_ingrid = isIngrid;

    std::cout << "[AnySample]: " << m_name << ", ID " << m_sampleid << std::endl
              << "[AnySample]: Bin edges: " << std::endl;
    for(int i = 0; i < m_D1edges.size(); ++i)
    {
        std::cout << m_D2edges[i].first << " " << m_D2edges[i].second << " "
                  << m_D1edges[i].first << " " << m_D1edges[i].second << std::endl;
    }

    //Default binning choices
    nbins_enu = 28;
    bins_enu = new double[nbins_enu + 1];
    for(int i=0;i<=nbins_enu;i++)
    {
        if(i<10) bins_enu[i] = i*0.1;
        else if(i>=10 && i<18) bins_enu[i] = 1.0 + (i-10)*0.2;
        else if(i==18) bins_enu[i] = 2.7;
        else if(i>=19 && i<23) bins_enu[i] = 3.0 + (i-19)*0.5;
        else if(i>=23) bins_enu[i] = 5.0 + (i-23)*1.0;
    }

    // New way of choosing default binning based on input binning,
    // relies on later bins always having larger values than earlier bins!

    int nbins_temp = v_d1edges.size();
    vector<double> bins_d1_vector;
    for(int i=0;i<nbins_temp;i++){
        if (bins_d1_vector.size()==0)
            bins_d1_vector.push_back(v_d1edges[i].first);
        else{
            if (bins_d1_vector.back()!=v_d1edges[i].first){
                if (!m_BufferBin && i>0 && v_d1edges[i].first<v_d1edges[i-1].first)
                    bins_d1_vector.push_back(v_d1edges[i-1].second);
                bins_d1_vector.push_back(v_d1edges[i].first);
            }
        }
    }
    bins_d1_vector.push_back(v_d1edges[nbins_temp-1].second);
    nbins_D1 = bins_d1_vector.size()-1;
    std::cout << "There are " << nbins_D1 << " d1 bins" << std::endl;

    bins_D1 = new double[nbins_D1 + 1];
    bool dqPlotBinsSet=false;
    for(int i=0;i<=nbins_D1;i++){
        if(i!=0){
            if(bins_d1_vector[i-1]>bins_d1_vector[i] && !dqPlotBinsSet){
                nbinsD1_toPlot=i-1;
                dqPlotBinsSet=true;
            }
        }
        bins_D1[i]  =  bins_d1_vector[i];
        cout << "bins_D1 " << i << " is " << bins_d1_vector[i] << endl;
    }
    if(!dqPlotBinsSet) nbinsD1_toPlot=nbins_D1;
    bins_D1toPlot = new double[nbinsD1_toPlot + 1];
    cout << "There are " << nbinsD1_toPlot << " d1 bins that will be used for plotting" << endl;

    for(int i=0;i<=nbinsD1_toPlot;i++){
        bins_D1toPlot[i]  =  bins_d1_vector[i];
        cout << "bins_D1toPlot " << i << " is " << bins_d1_vector[i] << endl;
    }

    nbins_temp = v_d1edges.size();
    vector<double> bins_d2_vector;
    for(int i=0;i<nbins_temp;i++){
        if (bins_d2_vector.size()==0) bins_d2_vector.push_back(v_d2edges[i].first);
        else{
            if (bins_d2_vector.back()!=v_d2edges[i].first) bins_d2_vector.push_back(v_d2edges[i].first);
        }
    }
    bins_d2_vector.push_back(v_d2edges[nbins_temp-1].second);

    nbins_D2 = bins_d2_vector.size()-1;
    cout << "There are " << nbins_D2 << " d2 bins" << endl;

    bins_D2 = new double[nbins_D2 + 1];

    for(int i=0;i<=nbins_D2;i++){
        bins_D2[i]  =  bins_d2_vector[i];
        cout << "bins_D2 " << i << " is " << bins_d2_vector[i] << endl;
    }

    nAnybins=m_D1edges.size();
    bins_Any = new double[nAnybins+1];
    for (int i=0; i<=nAnybins; i++){
        bins_Any[i]=i;
    }
    cout<<"Any bins defined"<<endl;
    //event distribution histo
    m_hpred = nullptr;
    m_hmc   = nullptr;
    m_hmc_true = nullptr;
    m_hsig   = nullptr;
    MakeHistos(); //with default binning

    std::cout << "[AnySample]: MakeHistos called." << std::endl;
    //data (or toy) histo
    m_hdata = nullptr;
    m_norm  = 1.0;
}

// dtor
AnySample::~AnySample()
{
    if(m_hpred != nullptr)
        m_hpred -> Delete();
    if(m_hmc != nullptr)
        m_hmc -> Delete();
    if(m_hmc_true != nullptr)
        m_hmc_true -> Delete();
    if(m_hdata != nullptr)
        m_hdata -> Delete();
    if(m_hsig != nullptr)
        m_hsig -> Delete();

    delete [] bins_D1;
    delete [] bins_D2;
    delete [] bins_enu;
}

// MakeEventHisto
void AnySample::MakeHistos()
{
    if(m_hpred != nullptr) m_hpred->Delete();
    m_hpred = new TH1D(Form("%s_pred_recD1D2", m_name.c_str()),
            Form("%s_pred_recD1D2", m_name.c_str()),
            nAnybins, bins_Any);
    m_hpred->SetDirectory(0);

    if(m_hmc != nullptr) m_hmc->Delete();
    m_hmc = new TH1D(Form("%s_mc_recD1D2", m_name.c_str()),
            Form("%s_mc_recD1D2", m_name.c_str()),
            nAnybins, bins_Any);
    m_hmc->SetDirectory(0);

    if(m_hmc_true != nullptr) m_hmc_true->Delete();
    m_hmc_true = new TH1D(Form("%s_mc_TrueD1D2", m_name.c_str()),
            Form("%s_mc_TrueD1D2", m_name.c_str()),
            nAnybins, bins_Any);
    m_hmc_true->SetDirectory(0);

    std::cout << "[AnySample]: " << nAnybins << " bins inside MakeHistos(). " << std::endl;
}

void AnySample::SetData(TObject *hdata)
{
    //clone the data histogram internally
    if(m_hdata != nullptr) m_hdata->Delete();
    m_hdata = (TH1D*)hdata->Clone(Form("%s_data", m_name.c_str()));
    m_hdata->SetDirectory(0);
}

// SetD1Binning
void AnySample::SetD1Binning(int nbins, double *bins)
{
    nbins_D1 = nbins;
    delete [] bins_D1;
    bins_D1 = new double[nbins_D1 + 1];
    for(int i=0;i<=nbins_D1;i++) bins_D1[i] = bins[i];
}

// SetD2Binning
void AnySample::SetD2Binning(int nbins, double *bins)
{
    nbins_D2 = nbins;
    delete [] bins_D2;
    bins_D2 = new double[nbins_D2 + 1];
    for(int i=0;i<=nbins_D2;i++) bins_D2[i] = bins[i];
}

// SetEnuBinning
void AnySample::SetEnuBinning(int nbins, double *bins)
{
    nbins_enu = nbins;
    delete [] bins_enu;
    bins_enu = new double[nbins_enu + 1];
    for(int i=0;i<=nbins_enu;i++) bins_enu[i] = bins[i];
}

int AnySample::GetAnyBinIndex(const double D1, const double D2)
{
    for(int i = 0; i < nAnybins; ++i)
    {
        if(D1 >= m_D1edges[i].first && D1 < m_D1edges[i].second)
        {
            if(D2 >= m_D2edges[i].first && D2 < m_D2edges[i].second)
            {
                return i;
            }
        }
    }
    return -1;
}

// FillEventHist
void AnySample::FillEventHisto(int datatype)
{
    if(m_empty)
        return; // This sample will have no events
    if(m_hpred != nullptr)
        m_hpred -> Reset();
    if(m_hmc != nullptr)
        m_hmc -> Reset();

    for(std::size_t i = 0; i < m_events.size(); ++i)
    {
        double D1_rec  = m_events[i].GetRecD1();
        double D2_rec  = m_events[i].GetRecD2();
        double D1_true = m_events[i].GetTrueD1();
        double D2_true = m_events[i].GetTrueD2();
        double wght    = m_events[i].GetEvWght();

        int anybin_index_rec = GetAnyBinIndex(D1_rec, D2_rec);
        int anybin_index_true = GetAnyBinIndex(D1_true, D2_true);

        m_hpred -> Fill(anybin_index_rec + 0.5, wght);
        m_hmc -> Fill(anybin_index_rec + 0.5, wght);
        m_hmc_true -> Fill(anybin_index_true + 0.5, wght);
    }

    m_hpred -> Scale(m_norm);
    m_hmc -> Scale(m_norm);

    //data without stat variation: useful when nuisance parameters
    //varied in the toys
    if(datatype == 1)
    {
        SetData(m_hpred);
        m_hdata->Reset();

        for(int j = 1; j <= m_hpred -> GetNbinsX(); ++j)
        {
            double val = m_hpred -> GetBinContent(j);
            if(val == 0.0)
            {
                std::cout << "[WARNING] In AnySample::FillEventHisto()\n"
                          << "[WARNING] " << m_name << " bin " << j << " has 0 entries. This may cause a problem with chi2 computations." << std::endl;
                continue;
            }

            m_hdata -> SetBinContent(j, val);  //without statistical fluctuations
        }
    }

    //data with statistical variation
    //(used when no nuisance sampling but nuisances are fitted)
    else if(datatype == 3)
    {
        SetData(m_hpred);
        m_hdata->Reset();

        for(int j = 1; j <= m_hpred -> GetNbinsX(); ++j)
        {
            double val = m_hpred -> GetBinContent(j);
            if(val == 0.0)
            {
                std::cout << "[WARNING] In AnySample::FillEventHisto()\n"
                          << "[WARNING] " << m_name << " bin " << j << " has 0 entries. This may cause a problem with chi2 computations." << std::endl;
                continue;
            }

            double poisson_val = gRandom->Poisson(val);
            m_hdata -> SetBinContent(j, poisson_val); //with statistical fluctuations
        }
    }

    //data from external (fake) dataset
    else if(datatype == 2 || datatype == 4)
    {
        SetData(m_hpred);
        m_hdata->Reset();

        float D1_rec_tree, D2_rec_tree, wght;
        int cut_branch;

        m_data_tree -> SetBranchAddress("cutBranch",&cut_branch);
        m_data_tree -> SetBranchAddress("weight",&wght);
        m_data_tree -> SetBranchAddress("D1Rec",&D1_rec_tree);
        m_data_tree -> SetBranchAddress("D2Rec",&D2_rec_tree);

        long int n_entries = m_data_tree -> GetEntries();
        for(std::size_t i = 0; i < n_entries; ++i)
        {
            m_data_tree -> GetEntry(i);
            if(cut_branch != m_sampleid)
                continue;

            for(int j = 0; j < nAnybins; ++j)
            {
                int anybin_index = GetAnyBinIndex(D1_rec_tree, D2_rec_tree);
                if(anybin_index != -1)
                {
                    m_hdata -> Fill(anybin_index + 0.5, wght);
                    break;
                }
                else
                {
                    std::cout << "[WARNING] In AnySample::FillEventHisto()\n"
                              << "[WARNING] No bin for current data event.\n"
                              << "[WARNING] D1_rec_tree: " << D1_rec_tree << std::endl
                              << "[WARNING] D2_rec_tree: " << D2_rec_tree << std::endl
                              << "[WARNING] CutBranch  : " << cut_branch << std::endl;
                    break;
                }
            }
        }

        std::cout << "[AnySample] Data histogram filled: " << std::endl;
        m_hdata -> Print();

        if(datatype == 4)
        {
            //Reweight fake data set
            //add MC or data (!!!!) statistical variations also to genie dataset to evaluate genie MC stat uncert
            //DON'T USE FOR REAL DATA!!!!!!!!!!!!

            std::cout << "[WARNING] REWEIGHTING DATA!" << std::endl;
            for(int j = 1; j <= m_hdata -> GetNbinsX(); ++j)
            {
                double val = m_hdata -> GetBinContent(j);
                if(val == 0.0)
                {
                    std::cout << "[WARNING] In AnySample::FillEventHisto()\n"
                              << "[WARNING] " << m_name << " bin " << j << " has 0 entries. This may cause a problem with chi2 computations." << std::endl;
                    continue;
                }
                double poisson_val = gRandom -> Poisson(val);
                m_hdata -> SetBinContent(j, poisson_val);  //add statistical fluctuations
            }
        }
    }
}

double AnySample::CalcChi2()
{
    if(m_empty == true)
        return 0.0;

    if(m_hdata == nullptr)
    {
        std::cerr << "[ERROR]: In AnySample::CalcChi2()\n"
                  << "[ERROR]: Need to define data histogram." << std::endl;
        return 0.0;
    }

    int nbins = m_hpred -> GetNbinsX();
    if(nbins != m_hdata -> GetNbinsX())
    {
        std::cerr << "[ERROR]: In AnySample::CalcChi2()\n"
                  << "[ERROR]: Binning mismatch between data and mc.\n"
                  << "[ERROR]: MC bins: " << nbins << ", Data bins: " << m_hdata -> GetNbinsX() << std::endl;
        return 0.0;
    }

    double chi2 = 0.0;
    for(int j = 1 ; j <= nbins; ++j)
    {
        double obs = m_hdata -> GetBinContent(j);
        double exp = m_hpred -> GetBinContent(j);
        if(exp > 0.0)
        {
            //added when external fake datasets (you cannot reweight when simply 0)
            // this didn't happen when all from same MC since if exp=0 then obs =0

            chi2 += 2 * (exp - obs);
            if(obs > 0.0)
                chi2 += 2 * obs * TMath::Log(obs/exp);

            if(chi2 < 0.0)
            {
                std::cerr << "[WARNING]: In AnySample::CalcChi2()\n"
                          << "[WARNING]: Stat chi2 is less than 0: " << chi2 << ", setting to 0." << std::endl;
                std::cerr << "[WARNING]: exp and obs is: " << exp << " and " << obs << "." << std::endl;
                chi2 = 0.0;
            }
        }
    }

    if(chi2 != chi2)
    {
        std::cerr << "[WARNING]: In AnySample::CalcChi2()\n"
                  << "[WARNING]: Stat chi2 is nan, setting to 0." << std::endl;
        chi2 = 0.0;
    }

    return chi2;
}

// GetSampleBreakdown
void AnySample::GetSampleBreakdown(TDirectory *dirout, const std::string& tag, bool save)
{
    int nreac = 8;
    const char *names[] = {"cc0pi0p", "cc0pi1p", "cc0pinp", "cc1pi+", "ccother",
        "backg", "Null", "OOFV"};
    TH1D *henu_rec[nreac];
    TH1D *hD1_rec[nreac];
    TH1D *hD2_rec[nreac];
    TH2D *hD1D2_rec[nreac];
    TH1D *hD1_true[nreac];
    TH1D *hD2_true[nreac];
    TH1D *hAnybin_true[nreac];
    TH1D *hAnybin_rec[nreac];
    int compos[nreac];

    //cout<<"AnySample::GetSampleBreakdown - Inializing histos of reactions" << endl;

    for(int i=0;i<nreac;i++)
    {
        compos[i] = 0;
        henu_rec[i] = new TH1D(Form("%s_RecEnu_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_RecEnu_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbins_enu, bins_enu);
        henu_rec[i]->SetDirectory(0);
        henu_rec[i]->GetXaxis()->SetTitle("Recon E_{#nu} (GeV)");

        //cout << "RecD1:" << endl;

        hD1_rec[i] = new TH1D(Form("%s_RecD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_RecD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot);
        hD1_rec[i]->SetDirectory(0);
        hD1_rec[i]->GetXaxis()->SetTitle("Recon D1 (GeV/c)");

        //cout << "RecD2:" << endl;

        hD2_rec[i] = new TH1D(Form("%s_RecD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_RecD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbins_D2, bins_D2);
        hD2_rec[i]->SetDirectory(0);
        hD2_rec[i]->GetXaxis()->SetTitle("Recon D2");

        //cout << "RecD1D2:" << endl;

        hD1D2_rec[i] = new TH2D(Form("%s_RecD1D2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_RecD1D2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot, nbins_D2, bins_D2);
        hD1D2_rec[i]->SetDirectory(0);
        hD1D2_rec[i]->GetXaxis()->SetTitle("Recon D1 (GeV/c)");
        hD1D2_rec[i]->GetXaxis()->SetTitle("Recon D2");

        //cout << "TrueD1" << endl;

        hD1_true[i] = new TH1D(Form("%s_TrueD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_TrueD1_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot);
        hD1_true[i]->SetDirectory(0);
        hD1_true[i]->GetXaxis()->SetTitle("True D1 (GeV/c)");

        //cout << "TrueD2" << endl;

        hD2_true[i] = new TH1D(Form("%s_TrueD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                Form("%s_TrueD2_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nbins_D2, bins_D2);
        hD2_true[i]->SetDirectory(0);
        hD2_true[i]->GetXaxis()->SetTitle("True D2");

        //cout << "TrueAny" << endl;

        hAnybin_true[i] = new TH1D(Form("%s_Anybins_true_%s_%s",   m_name.c_str(),names[i],tag.c_str()),
                Form("%s_Anybins_true_%s_%s",   m_name.c_str(),names[i],tag.c_str()),
                nAnybins, bins_Any);
        hAnybin_true[i]->SetDirectory(0);
        hAnybin_true[i]->GetXaxis()->SetTitle("Any bins");

        //cout << "RecAny" << endl;

        hAnybin_rec[i] = new TH1D(Form("%s_Anybins_rec_%s_%s",  m_name.c_str(),names[i],tag.c_str()),
                Form("%s_Anybins_rec_%s_%s", m_name.c_str(),names[i],tag.c_str()),
                nAnybins, bins_Any);
        hAnybin_rec[i]->SetDirectory(0);
        hAnybin_rec[i]->GetXaxis()->SetTitle("Any bins");
    }

    if(m_hsig != nullptr) m_hsig->Delete();
    m_hsig = new TH1D(Form("%s_signalOnly_%s", m_name.c_str(),tag.c_str()),
            Form("%s_signalOnly_%s", m_name.c_str(),tag.c_str()),
            nAnybins, bins_Any);
    m_hsig->SetDirectory(0);

    //loop over the events

    //cout<<"AnySample::GetSampleBreakdown - Collecting events" << endl;
    int Ntot = GetN();
    for(size_t i=0;i<m_events.size();i++)
    {
        //cout<<"AnySample::GetSampleBreakdown - In event loop iteration " << i << " out of " << m_events.size() << endl;
        double enu_rec, D1_rec, D2_rec, D1_true, D2_true, wght;
        enu_rec = m_events[i].GetRecEnu();
        D1_rec = m_events[i].GetRecD1();
        D2_rec = m_events[i].GetRecD2();
        D1_true = m_events[i].GetTrueD1();
        D2_true = m_events[i].GetTrueD2();
        wght    = m_events[i].GetEvWght();
        int rtype = m_events[i].GetTopology();
        //cout<< "AnySample::GetSampleBreakdown - rtype is: " << rtype << endl;

        // Warning - hard code hack warning ahead:
        // My reaction variable is mectopology from CC0Pi HL2 v1r15 so:
        // 0 - cc0pi0p
        // 1 - cc0pi1p
        // 2 - cc0pinp
        // 3 - cc1pi
        // 4 - ccother
        // 5 - BKG (not numuCC)
        // 6 - Nothing at all (WHY!?!?!?)
        // 7 - OOFGDFV
        // So I hack in a fix to stop a nullptr reaction cat.


        //if((rtype==7)) rtype=6; //BKG is 5 then OOFV is 7, 6 is skipped causing array to overrun

        //cout<< "AnySample::GetSampleBreakdown - Event breakdown:" << endl;
        //m_events[i].Print();


        //cout<< "AnySample::GetSampleBreakdown - Filling histos" << endl;

        compos[rtype]++;
        henu_rec[rtype]->Fill(enu_rec, wght);
        hD1_rec[rtype]->Fill(D1_rec, wght);
        hD2_rec[rtype]->Fill(D2_rec, wght);
        hD1_true[rtype]->Fill(D1_true, wght);
        hD2_true[rtype]->Fill(D2_true, wght);
        hD1D2_rec[rtype]->Fill(D1_rec, D2_rec, wght);


        //cout<< "AnySample::GetSampleBreakdown - Filling histos with analysis binning" << endl;

        for(int j=0; j<nAnybins; j++){
            if((D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
                    (D2_true > m_D2edges[j].first) && (D2_true < m_D2edges[j].second)){
                hAnybin_true[rtype]->Fill(j+0.5,wght);
                break;
            }
        }
        for(int j=0; j<nAnybins; j++){
            if((D1_rec > m_D1edges[j].first) && (D1_rec < m_D1edges[j].second)  &&
                    (D2_rec > m_D2edges[j].first) && (D2_rec < m_D2edges[j].second)){
                hAnybin_rec[rtype]->Fill(j+0.5,wght);
                break;
            }
        }
        //********************************************
        // Warning: Hardcoded signal definition below:
        //********************************************
        for(int j=0; j<nAnybins; j++){
            if( (D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
                    (D2_true > m_D2edges[j].first) && (D2_true < m_D2edges[j].second)  &&
                    ( (rtype==1) || (rtype==2) ) ) {
                //if( (D1_true > m_D1edges[j].first) && (D1_true < m_D1edges[j].second)  &&
                //    (D2_true > -0.5) && (D2_true < 0.5)  && (rtype==1 || rtype == 2) ) {
                m_hsig->Fill(j+0.5,wght);
                break;
            }
            }
        }

        //cout<<"AnySample::GetSampleBreakdown - Wrapping up" << endl;

        dirout->cd();
        //tree->Write();
        //cout << "Scale Factor Is: " << m_norm << endl;
        //m_hsig->Print("all");
        m_hsig->Scale(m_norm);
        //m_hsig->Print("all");

        for(int i=0;i<nreac;i++)
        {
            henu_rec[i]->Scale(m_norm);
            hD1_rec[i]->Scale(m_norm);
            hD2_rec[i]->Scale(m_norm);
            hD1_true[i]->Scale(m_norm);
            hD2_true[i]->Scale(m_norm);
            hD1D2_rec[i]->Scale(m_norm);
            hAnybin_true[i]->Scale(m_norm);
            hAnybin_rec[i]->Scale(m_norm);

            if(save){
                henu_rec[i]->Write();
                hD1_rec[i]->Write();
                hD2_rec[i]->Write();
                hD1_true[i]->Write();
                hD2_true[i]->Write();
                hD1D2_rec[i]->Write();
                hAnybin_true[i]->Write();
                hAnybin_rec[i]->Write();
            }

            henu_rec[i]->Delete();
            hD1_true[i]->Delete();
            hD2_true[i]->Delete();
            hD1_rec[i]->Delete();
            hD2_rec[i]->Delete();
            hD1D2_rec[i]->Delete();
            hAnybin_true[i]->Delete();
            hAnybin_rec[i]->Delete();
        }
        if(save){
            cout<<"============> Sample "<<m_name<<" <============"<<endl;
            for(int j=0;j<nreac;j++)
                cout<<setw(10)<<names[j]<<setw(5)<<j<<setw(10)<<compos[j]
                    <<setw(10)<<(float)(compos[j])/Ntot*100.0<<"%"<<endl;
        }
}

void AnySample::Write(TDirectory *dirout, const char *bsname, int fititer)
{
    dirout->cd();
    m_hpred->Write(Form("%s_pred", bsname));
    m_hmc_true->Write(Form("%s_true", bsname));
    if(fititer==0){
        m_hmc->Write(Form("%s_mc", bsname));
        if(m_hdata != nullptr) m_hdata->Write(Form("%s_data", bsname));
    }
}

void AnySample::GetSampleBreakdown(TDirectory *dirout, const std::string& tag, const std::vector<std::string>& topology, bool save)
{
    const int ntopology = topology.size();
    int compos[ntopology];
    std::vector<TH1D> henu_rec;
    std::vector<TH1D> hD1_rec;
    std::vector<TH1D> hD2_rec;
    std::vector<TH1D> hD1_true;
    std::vector<TH1D> hD2_true;
    std::vector<TH2D> hD1D2_rec;
    std::vector<TH1D> hAnybin_rec;
    std::vector<TH1D> hAnybin_true;

    for(int i = 0; i < ntopology; ++i)
    {
        compos[i] = 0;
        henu_rec.emplace_back(TH1D(Form("%s_RecEnu_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 Form("%s_RecEnu_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                 nbins_enu, bins_enu));
        henu_rec[i].SetDirectory(0);
        henu_rec[i].GetXaxis() -> SetTitle("Recon E_{#nu} (GeV)");

        hD1_rec.emplace_back(TH1D(Form("%s_RecD1_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_RecD1_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot));
        hD1_rec[i].SetDirectory(0);
        hD1_rec[i].GetXaxis() -> SetTitle("Recon D1 (GeV/c)");

        hD2_rec.emplace_back(TH1D(Form("%s_RecD2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_RecD2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nbins_D2, bins_D2));
        hD2_rec[i].SetDirectory(0);
        hD2_rec[i].GetXaxis() -> SetTitle("Recon D2 (GeV/c)");

        hD1D2_rec.emplace_back(TH2D(Form("%s_RecD1D2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_RecD1D2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot, nbins_D2, bins_D2));
        hD1D2_rec[i].SetDirectory(0);
        hD1D2_rec[i].GetXaxis() -> SetTitle("Recon D1D2 (GeV/c)");

        hD1_true.emplace_back(TH1D(Form("%s_TrueD1_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_TrueD1_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nbinsD1_toPlot, bins_D1toPlot));
        hD1_true[i].SetDirectory(0);
        hD1_true[i].GetXaxis() -> SetTitle("True D1 (GeV/c)");

        hD2_true.emplace_back(TH1D(Form("%s_TrueD2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_TrueD2_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nbins_D2, bins_D2));
        hD2_true[i].SetDirectory(0);
        hD2_true[i].GetXaxis() -> SetTitle("True D2 (GeV/c)");

        hAnybin_rec.emplace_back(TH1D(Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_Anybins_rec_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nAnybins, bins_Any));
        hAnybin_rec[i].SetDirectory(0);
        hAnybin_rec[i].GetXaxis() -> SetTitle("Any bins");

        hAnybin_true.emplace_back(TH1D(Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                Form("%s_Anybins_true_%s_%s", m_name.c_str(), topology[i].c_str(), tag.c_str()),
                nAnybins, bins_Any));
        hAnybin_true[i].SetDirectory(0);
        hAnybin_true[i].GetXaxis() -> SetTitle("Any bins");
    }

    if(m_hsig != nullptr)
        m_hsig -> Delete();
    m_hsig = new TH1D(Form("%s_signalOnly_%s", m_name.c_str(), tag.c_str()),
            Form("%s_signalOnly_%s", m_name.c_str(), tag.c_str()),
            nAnybins, bins_Any);
    m_hsig -> SetDirectory(0);

    int Ntot = GetN();
    for(std::size_t i = 0; i < m_events.size(); ++i)
    {
        double enu_rec, D1_rec, D2_rec, D1_true, D2_true, wght;
        enu_rec = m_events[i].GetRecEnu();
        D1_rec  = m_events[i].GetRecD1();
        D2_rec  = m_events[i].GetRecD2();
        D1_true = m_events[i].GetTrueD1();
        D2_true = m_events[i].GetTrueD2();
        wght    = m_events[i].GetEvWght();
        int evt_topology = m_events[i].GetTopology();

        // Warning - hard code hack warning ahead:
        // My reaction variable is mectopology from CC0Pi HL2 v1r15 so:
        // 0 - cc0pi0p
        // 1 - cc0pi1p
        // 2 - cc0pinp
        // 3 - cc1pi
        // 4 - ccother
        // 5 - BKG (not numuCC)
        // 6 - Nothing at all (WHY!?!?!?)
        // 7 - OOFGDFV
        // So I hack in a fix to stop a nullptr reaction cat.

        //cout<< "AnySample::GetSampleBreakdown - Filling histos" << endl;

        compos[evt_topology]++;
        henu_rec[evt_topology].Fill(enu_rec, wght);
        hD1_rec[evt_topology].Fill(D1_rec, wght);
        hD2_rec[evt_topology].Fill(D2_rec, wght);
        hD1_true[evt_topology].Fill(D1_true, wght);
        hD2_true[evt_topology].Fill(D2_true, wght);
        hD1D2_rec[evt_topology].Fill(D1_rec, D2_rec, wght);

        //cout<< "AnySample::GetSampleBreakdown - Filling histos with analysis binning" << endl;

        int anybin_index_rec = GetAnyBinIndex(D1_rec, D2_rec);
        int anybin_index_true = GetAnyBinIndex(D1_true, D2_true);
        hAnybin_rec[evt_topology].Fill(anybin_index_rec + 0.5, wght);
        hAnybin_true[evt_topology].Fill(anybin_index_true + 0.5, wght);

        if(evt_topology == 1 || evt_topology == 2)
            m_hsig -> Fill(anybin_index_true + 0.5, wght);
    }

    dirout->cd();
    m_hsig->Scale(m_norm);

    for(int i = 0; i < ntopology; ++i)
    {
        henu_rec[i].Scale(m_norm);
        hD1_rec[i].Scale(m_norm);
        hD2_rec[i].Scale(m_norm);
        hD1_true[i].Scale(m_norm);
        hD2_true[i].Scale(m_norm);
        hD1D2_rec[i].Scale(m_norm);
        hAnybin_true[i].Scale(m_norm);
        hAnybin_rec[i].Scale(m_norm);

        if(save == true)
        {
            henu_rec[i].Write();
            hD1_rec[i].Write();
            hD2_rec[i].Write();
            hD1_true[i].Write();
            hD2_true[i].Write();
            hD1D2_rec[i].Write();
            hAnybin_true[i].Write();
            hAnybin_rec[i].Write();
        }
    }

    if(save == true)
    {
        std::cout << "[AnySample]: GetSampleBreakdown()\n"
                  << "============ Sample " << m_name << " ==========="<<endl;

        for(int j = 0; j < ntopology; ++j)
            std::cout << std::setw(10) << topology[j] << std::setw(5) << j
                      << std::setw(5) << compos[j] << std::setw(10)
                      << ((1.0 * compos[j]) / Ntot) * 100.0 << "%" << std::endl;
    }
}

