#include "XsecFitter.hh"

XsecFitter::XsecFitter(const int seed, const int num_threads)
{
    rng = new TRandom3(seed);
    //set gRandom to our rand
    gRandom = rng;
    m_fitter = nullptr;
    m_fcn   = nullptr;
    m_dir   = nullptr;
    m_freq  = -1;
    m_threads = num_threads;
}

// dtor
XsecFitter::~XsecFitter()
{
    m_dir = nullptr;
    if(rng != nullptr)
        delete rng;
    if(m_fitter != nullptr)
        delete m_fitter;
    if(m_fcn != nullptr)
        delete m_fcn;
}

// SetSeed
void XsecFitter::SetSeed(int seed)
{
    if(rng == nullptr)
    {
        rng = new TRandom3(seed);
        gRandom = rng; //Global pointer
    }
    else
        rng -> SetSeed(seed);
}

void XsecFitter::FixParameter(const std::string& par_name, const double& value)
{
    auto iter = std::find(par_names.begin(), par_names.end(), par_name);
    if(iter != par_names.end())
    {
        const int i = std::distance(par_names.begin(), iter);
        m_fitter -> SetVariable(i, par_names.at(i).c_str(), value, 0);
        m_fitter -> FixVariable(i);
        std::cout << "[XsecFitter]: Fixing parameter " << par_names.at(i) << " to value "
                  << value << std::endl;
    }
    else
    {
        std::cerr << "[Error]: In function XsecFitter::FixParameter()\n"
                  << "[Error]: Parameter " << par_name << " not found!" << std::endl;
    }
}

// PrepareFitter
void XsecFitter::InitFitter(std::vector<AnaFitParameters*> &fitpara, double reg, const std::string& paramVectorFname)
{
    paramVectorFileName = paramVectorFname;
    reg_p1 = reg;
    m_fitpara = fitpara;
    std::vector<double> par_step, par_low, par_high;

    m_npar = 0;
    m_nparclass.clear();
    //get the parameter info
    for(std::size_t i=0;i<m_fitpara.size();i++)
    {
        m_npar += m_fitpara[i] -> GetNpar();
        m_nparclass.push_back(m_fitpara[i]->GetNpar());

        std::vector<string> vec0;
        m_fitpara[i]->GetParNames(vec0);
        par_names.insert(par_names.end(), vec0.begin(), vec0.end());

        std::vector<double> vec1, vec2;
        m_fitpara[i]->GetParPriors(vec1);
        par_prefit.insert(par_prefit.end(), vec1.begin(), vec1.end());

        m_fitpara[i]->GetParSteps(vec1);
        par_step.insert(par_step.end(), vec1.begin(), vec1.end());

        m_fitpara[i]->GetParLimits(vec1, vec2);
        par_low.insert(par_low.end(), vec1.begin(), vec1.end());
        par_high.insert(par_high.end(), vec2.begin(), vec2.end());
    }

    if(m_npar == 0)
    {
        std::cerr << "[ERROR]: No fit parameters were defined." << std::endl;
        return;
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "           Initilizing fitter              " << std::endl;
    std::cout << "    Number of parameters = "  << m_npar << std::endl;
    std::cout << "===========================================" << std::endl;

    m_fitter = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
    m_fcn = new ROOT::Math::Functor(this, &XsecFitter::CalcLikelihood, m_npar);

    m_fitter -> SetFunction(*m_fcn);
    m_fitter -> SetPrintLevel(2);
    m_fitter -> SetMaxIterations(1E6);
    m_fitter -> SetTolerance(1E-4);

    for(int i = 0; i < m_npar; ++i)
    {
        m_fitter -> SetVariable(i, par_names[i], par_prefit[i], par_step[i]);
        m_fitter -> SetVariableLimits(i, par_low[i], par_high[i]);
    }

    // Save prefit parameters:
    prefitParams = new TH1D("prefitParams", "prefitParams", m_npar, 0, m_npar);
    int paramNo = 1;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        for(int j = 0; j < m_fitpara[i] -> GetNpar(); ++j){
            prefitParams->SetBinContent(paramNo, m_fitpara[i]->GetParPrior(j));
            if(m_fitpara[i]->HasCovMat() && (!(m_fitpara[i]->HasRegCovMat())) && (i!=0))
            {
                TMatrixDSym* covMat = m_fitpara[i]->GetCovarMat();
                prefitParams->SetBinError(paramNo, sqrt((*covMat)[j][j]));
            }
            else
                prefitParams->SetBinError(paramNo, 0);
            paramNo++;
        }
    }
}

// Fit
//datatype = 1 if toy from nuisances - syst variation
//           2 if external (fake) dataset
//           3 if stat fluctuations of nominal
//           4 if stat fluctuations of (fake) data
//           5 if toy from nuisances - syst + reg constrained fit variation
//           6 if toy from nuisances - syst + random fit variation
//           7 if toy from nuisances - syst + reg constrained fit variation, fit w/o reg
//           8 if asimov (data==MC)
//           9 if fake data from param vector
void XsecFitter::Fit(std::vector<AnaSample*> &samples, const std::vector<std::string>& topology, int datatype, int fitMethod, int statFluct)
{
    std::cout << "[XsecFitter]: Starting to fit." << std::endl;
    m_calls = 0;
    m_samples = samples;
    if(m_fitter == nullptr)
    {
        std::cerr << "[ERROR]: In XsecFitter::Fit()\n"
                  << "[ERROR]: Fitter has not been initialized." << std::endl;
        return;
    }

    //generate toy data to fit
    if(datatype==1 || datatype==5 || datatype==6 || datatype==7 || datatype==9){
        if(datatype==1) GenerateToyData(0,0,statFluct);
        if(datatype==5 || datatype==7) GenerateToyData(0,1,statFluct);
        if(datatype==6) GenerateToyData(0,2,statFluct);
        if(datatype==9) GenerateToyData(0,3,statFluct);
        cout<<"toy data generated"<<endl;
        //save hists if requested
        for(size_t s=0;s<m_samples.size();s++)
        {
            m_samples[s]->GetSampleBreakdown(m_dir, "thrown", topology, false);
        }
    }
    else if(datatype==2 || datatype==3 || datatype==4){
        for(size_t s=0;s<m_samples.size();s++)
        {
            m_samples[s]->FillEventHisto(datatype);
        }
    }
    else if(datatype==8 || datatype==10){
        for(size_t s=0;s<m_samples.size();s++)
        {
            m_samples[s]->FillEventHisto(1);
        }
    }
    else
    {
        std::cerr << "[ERROR]: In XsecFitter::Fit()\n"
                  << "[ERROR]: No valid fitting mode provided." << std::endl;
        return;
    }
    if(m_freq >= 0 && m_dir)
        DoSaveEvents(m_calls);

    //Do fit
    std::cout << "[XsecFitter]: Fit prepared." << std::endl;

    std::cout << "[XsecFitter]: Calling MIGRAD ..." << std::endl;
    m_fitter -> Minimize();
    std::cout << "[XsecFitter]: Calling HESSE ..." << std::endl;
    m_fitter -> Hesse();

    //fill chi2 info
    if(m_dir)
        DoSaveChi2();

    const int ndim  = m_fitter -> NDim();
    const int nfree = m_fitter -> NFree();
    double cov_array[ndim*ndim];
    m_fitter -> GetCovMatrix(cov_array);

    TMatrixDSym cov_matrix(ndim, cov_array);

    //Calculate Corrolation Matrix
    TMatrixDSym cor_matrix(ndim);
    for(int r = 0; r < ndim; ++r)
        for(int c = 0; c < ndim; ++c)
            cor_matrix[r][c] = cov_matrix[r][c] / std::sqrt(cov_matrix[r][r] * cov_matrix[c][c]);

    const double* par_val = m_fitter -> X();
    const double* par_err = m_fitter -> Errors();

    //save fit results
    TVectorD postfit_param(ndim);
    std::vector< std::vector<double> > res_pars;
    std::vector< std::vector<double> > err_pars;
    int k = 0;
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        std::vector<double> vec_res;
        std::vector<double> vec_err;

        for(int j=0;j<m_nparclass[i];j++)
        {
            vec_err.push_back(par_err[k]);
            double parvalue = par_val[k];
            vec_res.push_back(parvalue);
            postfit_param[k] = parvalue;
            k++;
        }

        res_pars.push_back(vec_res);
        err_pars.push_back(vec_err);
    }

    if(k != ndim)
    {
        std::cout << "[ERROR] Number of parameters." << std::endl;
        return;
    }

    m_dir->cd();
    cov_matrix.Write("res_cov_matrix");
    cor_matrix.Write("res_cor_matrix");
    postfit_param.Write("res_vector");
    prefitParams->Write("prefitParams");

    DoSaveResults(res_pars, err_pars);
    if(m_freq >= 0 && m_dir)
        DoSaveFinalEvents(m_calls, res_pars);
}

// GenerateToyData
// toytype = 0 -> Nuisance throws only
// toytype = 1 -> Also throw ci parmas from a reg prior
// toytype = 2 -> Also throw ci parmas from a flat prior
// toytype = 3 Generate toy data from provided paramVector
void XsecFitter::GenerateToyData(int toyindx, int toytype, int statFluct)
{
    //do parameter throws
    std::vector<std::vector<double> > par_throws;
    double chi2_sys = 0.0;
    double chi2_reg = 0.0;
    int gparamno = 0;
    TFile *finput = nullptr;;
    TVectorD *paramVec = nullptr;

    if(toytype==3){
        finput= new TFile(paramVectorFileName.c_str(),"READ");
        if(!finput) cout << "WARNING: Issue opening input param vector file" << endl;
        paramVec = ((TVectorD*)finput->Get("res_vector"));
        if(!paramVec) cout << "WARNING: Issue getting param vector fromfile" << endl;
    }

    for(size_t i=0;i<m_fitpara.size();i++)
    {
        cout << "XsecFitter::GenerateToyData fit param set: " << i << endl;
        //cout << "HasRegCovMat : " << m_fitpara[i]->HasRegCovMat() << endl;
        bool throwOkay=false;

        if(toyindx == 0) m_fitpara[i]->InitThrows();
        vector<double> pars;
        if      ((toytype==0 || toytype==2) && i!=0) m_fitpara[i]->DoThrow(pars, 0);
        else if ((toytype==0 || toytype==2) && i==0) m_fitpara[i]->DoThrow(pars, 3);
        if(toytype==1 && (m_fitpara[i]->HasRegCovMat())==false) m_fitpara[i]->DoThrow(pars, 1);

        if(toytype==1 && (m_fitpara[i]->HasRegCovMat())==true){
            cout << "Finding good set of params from throws: " << endl;
            while(throwOkay==false){
                m_fitpara[i]->DoThrow(pars, 1);
                throwOkay=true;
                cout << "Testing set: " << endl;
                for(size_t j=0;j<pars.size();j++){
                    cout << pars[j] << ", ";
                    if(pars[j]<0.3 || pars[j]>1.7) throwOkay=false;
                }
                cout << endl;
            }
            cout << "Good set found" << endl;
        }
        //m_fitpara[i]->GetParPriors(pars);

        if(toytype==2){
            //Throw fit params randomly in a sensible range (here 0.1 - 3.1, want to stay away from 0 boundary)
            if((m_fitpara[i]->HasCovMat())==false){
                for(size_t j=0;j<pars.size();j++){
                    pars[j] = (3*gRandom -> Uniform(0,1))+0.1;
                }
            }
        }

        if(toytype==3){
            for(int j=0; j<m_fitpara[i]->GetNpar(); j++){
                pars.push_back((*paramVec)(gparamno));
                gparamno++;
            }
        }

        par_throws.push_back(pars);
        chi2_sys += m_fitpara[i]->GetChi2(pars);
        if(i==0) chi2_reg += chi2_reg;
        cout << "Toy data generated with following values: " << endl;
        cout<<"Parameters "<<m_fitpara[i]->GetName()<<endl;
        for(size_t j=0;j<pars.size();j++) cout<<j<<": "<<pars[j]<< ", ";
        cout << endl;
    }
    vec_chi2_sys.push_back(chi2_sys);
    vec_chi2_reg.push_back(chi2_reg);

    if(toytype==3){
        if(paramVec->GetNrows() != gparamno) cout << "ERROR: mismatch in parameters required (" << gparamno << ") vs paramters in read vector (" << paramVec->GetNrows() << ")." << endl;
    }

    //Generate toy data histograms
    double chi2_stat;
    if(statFluct==0) chi2_stat=FillSamples(par_throws, 1);
    else if(statFluct==1) chi2_stat=FillSamples(par_throws, 3);
    vec_chi2_stat.push_back(chi2_stat);

    if(m_freq>=0  && m_dir) DoSaveParams(par_throws);
}


// FillSample with new parameters
//datatype = 0 if fit iteration
//           1 if toy dataset from nuisances
double XsecFitter::FillSamples(std::vector<std::vector<double> >& new_pars, int datatype)
{
    double chi2 = 0.0;
    bool output_chi2 = false;
    if((m_calls<1001 && (m_calls%100==0 || m_calls<20)) || (m_calls>1001 && m_calls%1000==0))
        output_chi2 = true;

    //loop over samples
    #pragma omp parallel for num_threads(m_threads)
    for(int s = 0; s < m_samples.size(); ++s)
    {
        //loop over events
        const unsigned int num_events = m_samples[s] -> GetN();
        const std::string det = m_samples[s] -> GetDetector();
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            //do weights for each AnaFitParameters obj
            for(int j = 0; j < m_fitpara.size(); ++j)
            {
                m_fitpara[j] -> ReWeight(ev, det, s, i, new_pars[j]);
            }
        }

        m_samples[s] -> FillEventHisto(datatype);
        double sample_chi2 = m_samples[s] -> CalcChi2();

        //calculate chi2 for each sample
        #pragma omp atomic
        chi2 += sample_chi2;

        if(output_chi2)
        {
            std::cout << "[XsecFitter]: Chi2 for sample " << m_samples[s] -> GetName() << " is "
                      <<  sample_chi2 << std::endl;
        }
    }

    return chi2;
}

double XsecFitter::CalcLikelihood(const double* par)
{
    m_calls++;

    bool output_chi2 = false;
    if((m_calls<1001 && (m_calls%100==0 || m_calls<20)) || (m_calls>1001 && m_calls%1000==0))
        output_chi2 = true;

    //Regularisation:
    int k=0;
    double chi2_reg = 0.0;
    double chi2_sys = 0.0;
    std::vector<std::vector<double> > new_pars;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        std::vector<double> vec;
        for(int j = 0; j < m_nparclass[i]; ++j)
        {
            vec.push_back(par[k++]);
            if(output_chi2)
                std::cout << "Param " << j << " of class " << i << " has value " << par[k-1] << std::endl; // << " giving total chi2_reg " << chi2_reg << endl;
        }

        chi2_sys += m_fitpara[i] -> GetChi2(vec);
        // "Systematic error" on the fit parameters comes from the regularisation covatriance matrix, store as a chi2 reg error:
        if(i == 0)
        {
            chi2_reg += chi2_sys;
            chi2_sys -= m_fitpara[i]->GetChi2(vec);
        }

        new_pars.push_back(vec);
        if(output_chi2)
            std::cout << "chi2_sys contribution from param set " << i << " is " << m_fitpara[i]->GetChi2(vec) << endl;
    }
    if(output_chi2)
        std::cout << "chi2_sys contribution from regularisation " << chi2_reg << endl;
    //chi2_sys += chi2_reg;
    vec_chi2_sys.push_back(chi2_sys);
    vec_chi2_reg.push_back(chi2_reg);

    double chi2_stat = FillSamples(new_pars, 0);
    vec_chi2_stat.push_back(chi2_stat);

    //save hists if requested
    if(m_calls % (m_freq*10000) == 0 && m_dir)
    {
        DoSaveParams(new_pars);
        if(m_calls % (m_freq*10000) == 0)
            DoSaveEvents(m_calls);
    }

    //Print status of the fit:
    if(output_chi2)
    {
        std::cout << "m_calls is: " << m_calls << endl;
        std::cout << "Chi2 for this iter: " << chi2_stat + chi2_sys + chi2_reg << endl;
        std::cout << "Chi2 stat / syst / reg : " << chi2_stat << " / " << chi2_sys << " / " << chi2_reg << std::endl;
    }

    return chi2_stat + chi2_sys + chi2_reg;
    // Update final chi2 file for L curve calc:

    // ofstream outfile;
    // outfile.open("chi2.txt", ios::out | ios::trunc );
    // outfile << reg_p1 << ", " << chi2_stat << ", " << chi2_reg << ", " << chi2_sys << ", " << chi2_stat + chi2_sys << endl;
    // outfile.close();
}

// Write hists for reweighted events
void XsecFitter::DoSaveEvents(int fititer)
{
    for(size_t s=0;s<m_samples.size();s++){
        m_samples[s]->Write(m_dir, Form("evhist_sam%d_iter%d",(int)s,m_calls), fititer);
    }
}

// Write hists for reweighted events
void XsecFitter::DoSaveFinalEvents(int fititer, std::vector<std::vector<double> > res_params)
{
    std::cout << std::endl << "**************************" << std::endl
              << "Fit finished, saving results .. " << std::endl
              << "**************************" << std::endl;

    outtree = new TTree("selectedEvents", "selectedEvents");
    InitOutputTree();
    for(size_t s=0;s<m_samples.size();s++){
        m_samples[s]->Write(m_dir, Form("evhist_sam%d_finaliter",(int)s), fititer);
        //Event loop:
        //cout << "Saving reweighted event tree ..." << endl;
        for(int i=0;i<m_samples[s]->GetN();i++)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            for(size_t j=0;j<m_fitpara.size();j++)
            {
                if(((TString)(m_fitpara[j]->GetName())).Contains("par_detFine")) continue;
                std::string det = m_samples[s] -> GetDetector();
                m_fitpara[j]->ReWeight(ev, det, s, i, res_params[j]);
            }

            cutBranch = ev->GetSampleType();
            D1true = ev->GetTrueD1();
            D2true = ev->GetTrueD2();
            mectopology = ev->GetTopology();
            reaction = ev->GetReaction();
            D1Reco = ev->GetRecD1();
            D2Reco = ev->GetRecD2();
            weightNom = (ev->GetEvWght()) * (ev->GetEvWghtMC());
            weightMC = ev->GetEvWghtMC();
            weight = ev->GetEvWght() * m_potratio;
            pMomRec = ev->GetpMomRec();
            pMomTrue = ev->GetpMomTrue();
            muMomRec = ev->GetmuMomRec();
            muMomTrue = ev->GetmuMomTrue();
            muCosThetaRec = ev->GetmuCosThetaRec();
            muCosThetaTrue = ev->GetmuCosThetaTrue();
            pCosThetaRec = ev->GetpCosThetaRec();
            pCosThetaTrue = ev->GetpCosThetaTrue();
            outtree->Fill();
        }
    }
    m_dir->cd();
    outtree->Write();
}


// Write hists for parameter values
void XsecFitter::DoSaveParams(vector< vector<double> > new_pars)
{
    //loop on number of parameter classes
    vector<double> paramVec;
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        TH1D* histopar = new TH1D(Form("paramhist_par%s_iter%d",m_fitpara[i]->GetName().c_str(),m_calls),
                Form("paramhist_par%s_iter%d",m_fitpara[i]->GetName().c_str(),m_calls),
                m_nparclass[i], 0, m_nparclass[i]);
        //loop on number of parameters in each class
        vector <string> parnames;
        m_fitpara[i]->GetParNames(parnames);
        for(int j=0;j<m_nparclass[i];j++)
        {
            histopar->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histopar->SetBinContent(j+1,(new_pars[i])[j]);
            paramVec.push_back((new_pars[i])[j]);
        }
        m_dir->cd();

        histopar->Write();
        delete histopar;
    }
    //Convert into TVector
    TVectorD TParamVec(paramVec.size());
    for(int i=0;i<(paramVec.size());i++) TParamVec(i)=paramVec[i];

    //Write:
    //cout << "****************************" << endl;
    //TParamVec.Print();
    TParamVec.Write(Form("paramVector_iter%d",m_calls));
}

void XsecFitter::DoSaveChi2()
{
    TH1D* histochi2stat = new TH1D("chi2_stat_periter","chi2_stat_periter", m_calls+1, 0, m_calls+1);
    TH1D* histochi2sys = new TH1D("chi2_sys_periter","chi2_sys_periter", m_calls+1, 0, m_calls+1);
    TH1D* histochi2reg = new TH1D("chi2_reg_periter","chi2_reg_periter", m_calls+1, 0, m_calls+1);
    TH1D* histochi2tot = new TH1D("chi2_tot_periter","chi2_tot_periter", m_calls+1, 0, m_calls+1);

    TH1D* reg_param = new TH1D("reg_param","reg_param", m_calls+1, 0, m_calls+1);
    TH1D* reg_param2 = new TH1D("reg_param2","reg_param2", m_calls+1, 0, m_calls+1);
    reg_param->Fill(reg_p1);

    if(vec_chi2_stat.size() != vec_chi2_sys.size()){
        cout<<"Number of saved iterations for chi2 stat and chi2 syst different -  EXITING"<<endl;
        exit(-1);
    }
    //loop on number of parameter classes
    for(size_t i=0;i<vec_chi2_stat.size();i++)
    {
        histochi2stat->SetBinContent(i+1,vec_chi2_stat[i]);
        histochi2sys->SetBinContent(i+1,vec_chi2_sys[i]);
        histochi2reg->SetBinContent(i+1,vec_chi2_reg[i]);
        histochi2tot->SetBinContent(i+1,vec_chi2_sys[i] + vec_chi2_stat[i]);
    }
    m_dir->cd();
    histochi2stat->Write();
    histochi2sys->Write();
    histochi2reg->Write();
    histochi2tot->Write();
    reg_param->Write();
    reg_param2->Write();

    delete histochi2stat;
    delete histochi2sys;
    delete histochi2reg;
    delete histochi2tot;
}

void XsecFitter::DoSaveResults(std::vector<std::vector<double> >& parresults, std::vector<std::vector<double> >& parerrors)
{
    //loop on number of parameter classes
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        TH1D* histopar = new TH1D(Form("paramhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerr = new TH1D(Form("paramerrhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);

        //loop on number of parameters in each class
        vector <string> parnames;
        m_fitpara[i]->GetParNames(parnames);
        for(int j=0;j<m_nparclass[i];j++)
        {
            histopar->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histopar->SetBinContent(j+1,(parresults[i])[j]);
            histoparerr->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histoparerr->SetBinContent(j+1,(parerrors[i])[j]);
        }

        m_dir->cd();
        histopar->Write();
        histoparerr->Write();

        delete histopar;
        delete histoparerr;
    }

    std::vector<std::string> topology = {"cc0pi0p", "cc0pi1p", "cc0pinp", "cc1pi+", "ccother",
        "backg", "Null", "OOFV"};

    //ADDED save actual final results
    for(size_t s=0;s<m_samples.size();s++)
    {
        m_samples[s]->GetSampleBreakdown(m_dir, "fit", topology, false);
    }
}
