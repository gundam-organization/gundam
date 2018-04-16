#include "XsecFitter.hh"

// ClassImp(XsecFitter);

TVirtualFitter *fitter = 0;

//dummy function for SetFCN
void dummy_fcn( Int_t &npar, Double_t *gin, Double_t &f,
        Double_t *par, Int_t iflag )
{
    XsecFitter *myObj;
    myObj = dynamic_cast<XsecFitter*>(fitter->GetObjectFit());
    //Call the actual chi2 function inside our class
    myObj->fcn( npar, gin, f, par, iflag );
}

// ctor
XsecFitter::XsecFitter(int seed)
{
    rng = new TRandom3(seed);
    //set gRandom to our rand
    gRandom = rng;
    m_dir   = nullptr;
    m_freq  = -1;
}

// dtor
XsecFitter::~XsecFitter()
{
    m_dir = nullptr;
    if(rng != nullptr)
        delete rng;
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
        fitter -> SetParameter(i, par_names.at(i).c_str(), value, 0, value, value);
        fitter -> FixParameter(i);
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
void XsecFitter::InitFitter(std::vector<AnaFitParameters*> &fitpara, double reg, double reg2, int nipsbinsin, const std::string& paramVectorFname)
{
    paramVectorFileName=paramVectorFname;
    reg_p1=reg;
    reg_p2=reg2;
    m_fitpara = fitpara;
    nipsbins = nipsbinsin;
    std::vector<double> par_step, par_low, par_high;

    m_npar = 0;
    m_nparclass.clear();
    //get the parameter info
    for(std::size_t i=0;i<m_fitpara.size();i++)
    {
        m_npar += (int)m_fitpara[i]->Npar;
        m_nparclass.push_back((int)m_fitpara[i]->Npar);

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
        fitter = 0;
        return;
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "           Initilizing fitter              " << std::endl;
    std::cout << "    Number of parameters = "  << m_npar << std::endl;
    std::cout << "===========================================" << std::endl;


    TVirtualFitter::SetDefaultFitter("Minuit");
    fitter = TVirtualFitter::Fitter(0, m_npar);
    //pass our object to the fitter
    fitter->SetObjectFit(this);
    //set the dummy fcn
    fitter->SetFCN(dummy_fcn);

    double arglist[5];
    // use this to have low printout
    // arglist[0] = -1.;
    arglist[0] = 1;
    fitter->ExecuteCommand("SET PRINT", arglist, 1);

    //init fitter stuff
    for(int i=0;i<m_npar;i++)
    {
        fitter->SetParameter(i, par_names[i].c_str(), par_prefit[i], par_step[i], par_low[i], par_high[i]);
    }

    // Save prefit parameters:

    prefitParams = new TH1D("prefitParams", "prefitParams", m_npar, 0, m_npar);
    int paramNo=1;
    for(std::size_t i=0;i<m_fitpara.size();i++)
    {
        for(int j=0; j<(int)m_fitpara[i]->Npar; j++){
            prefitParams->SetBinContent(paramNo, m_fitpara[i]->GetParPrior(j));
            if(m_fitpara[i]->HasCovMat() && (!(m_fitpara[i]->HasRegCovMat())) && (i!=0)){
                TMatrixDSym* covMat = m_fitpara[i]->GetCovarMat();
                prefitParams->SetBinError(paramNo, sqrt((*covMat)[j][j]));
            }
            else prefitParams->SetBinError(paramNo, 0);
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
    std::cout << "Starting to fit." << std::endl;
    m_calls = 0;
    m_samples = samples;
    if(!fitter)
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
            m_samples[s]->GetSampleBreakdown(m_dir, "thrown", topology, true);
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

    //Collect Sample Histos
    CollectSampleHistos();

    //Do fit
    std::cout << "[XsecFitter]: Fit prepared." << std::endl;
    double arglist[5];
    //arglist[0] = 1000; //number of calls
    arglist[0] = 1000000; //number of calls
    //arglist[1] = 1.0E-4; //tolerance
    arglist[1] = 1.0E-4; //tolerance

    if(fitMethod==1){
        std::cout << "[XsecFitter]: Calling MIGRAD ..." << std::endl;
        fitter->ExecuteCommand("MIGRAD", arglist, 2);
    }
    else if(fitMethod==2){
        std::cout << "[XsecFitter]: Calling MIGRAD ..." << std::endl;
        fitter->ExecuteCommand("MIGRAD", arglist, 2);
        std::cout << "[XsecFitter]: Calling HESSE ..." << std::endl;
        fitter->ExecuteCommand("HESSE", arglist, 2);
    }
    else if(fitMethod==3){
        std::cout << "[XsecFitter]: Calling MINOS ..." << std::endl;
        fitter->ExecuteCommand("MINOS", arglist, 2);
    }

    //print the results
    Double_t amin;
    Double_t edm, errdef;
    Int_t nvpar, nparx;
    fitter->GetStats(amin, edm, errdef, nvpar, nparx);

    //fill chi2 info
    if(m_dir) DoSaveChi2();

    //Get Error Matrix
    TMatrixDSym matrix(nvpar,fitter->GetCovarianceMatrix());

    //Calculate Corrolation Matrix
    TMatrixD cormatrix(matrix.GetNrows(),matrix.GetNcols());
    for(int r=0;r<matrix.GetNrows();r++){
        for(int c=0;c<matrix.GetNcols();c++){
            cormatrix[r][c]= matrix[r][c]/sqrt((matrix[r][r]*matrix[c][c]));
        }
    }

    //save fit results
    TVectorD fitVec(nparx);
    vector< vector<double> > res_pars;
    vector< vector<double> > err_pars;
    vector< vector<double> > errplus_pars;
    vector< vector<double> > errminus_pars;
    vector< vector<double> > errpara_pars;
    vector< vector<double> > errglobc_pars;
    vector< vector<double> > errprof_pars;
    Double_t errplus, errminus, errpara, errglobc, profErr;
    int k=0;
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        vector<double> vec_res;
        vector<double> vec_err;
        vector<double> vec_errplus;
        vector<double> vec_errminus;
        vector<double> vec_errpara;
        vector<double> vec_errglobc;
        vector<double> vec_errprof;
        for(int j=0;j<m_nparclass[i];j++){
            vec_err.push_back(fitter->GetParError(k));
            double parvalue=fitter->GetParameter(k);
            vec_res.push_back(parvalue);
            fitVec(k) = parvalue;

            // Get full errors:
            fitter->GetErrors(k, errplus, errminus, errpara, errglobc);
            vec_errplus.push_back(errplus);
            vec_errminus.push_back(errminus);
            vec_errpara.push_back(errpara);
            vec_errglobc.push_back(errglobc);

            // Get profiled error:
            //WIP - it works but is a bit of an aproximation, use at your own risk
            /*
               profErr = FindProfiledError(k, matrix);
               vec_errprof.push_back(profErr);
               cout << "Profiled Error for param " << k << " is " << profErr << endl;
               */
            vec_errprof.push_back(1.0);

            k++;
        }
        res_pars.push_back(vec_res);
        err_pars.push_back(vec_err);
        errplus_pars.push_back(vec_errplus);
        errminus_pars.push_back(vec_errminus);
        errpara_pars.push_back(vec_errpara);
        errglobc_pars.push_back(vec_errglobc);
        errprof_pars.push_back(vec_errprof);
    }



    if(k != nparx)
    {
        std::cout << "[ERROR] Number of parameters." << std::endl;
        return;
    }
    m_dir->cd();
    matrix.Write("res_cov_matrix");
    cormatrix.Write("res_cor_matrix");
    fitVec.Write("res_vector");
    prefitParams->Write("prefitParams");

    DoSaveResults(res_pars,err_pars,errplus_pars,errminus_pars,errpara_pars,errglobc_pars,errprof_pars,amin);
    if(m_freq>=0 && m_dir) DoSaveFinalEvents(m_calls, res_pars);
}

// GenerateToyData
// toytype = 0 -> Nuisance throws only
// toytype = 1 -> Also throw ci parmas from a reg prior
// toytype = 2 -> Also throw ci parmas from a flat prior
// toytype = 3 Generate toy data from provided paramVector
void XsecFitter::GenerateToyData(int toyindx, int toytype, int statFluct)
{
    //do parameter throws
    vector< vector<double> > par_throws;
    double chi2_sys = 0.0;
    double chi2_reg = 0.0;
    int gparamno=0;
    TFile *finput;
    TVectorD *paramVec;

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
            for(int j=0; j<m_fitpara[i]->Npar; j++){
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
double XsecFitter::FillSamples(vector< vector<double> > new_pars, int datatype)
{
    double chi2 = 0.0;
    //cout << "nfitpar is: " << m_fitpara.size() << endl;

    bool isGonnaBeABiggun = false;
    if((m_calls<1001 && (m_calls%100==0 || m_calls<20)) || (m_calls>1001 && m_calls%1000==0) || (m_calls>10001 && m_calls%10000==0)) isGonnaBeABiggun = true;

    //loop over samples
    for(size_t s=0;s<m_samples.size();s++)
    {
        //loop over events
        for(int i=0;i<m_samples[s]->GetN();i++)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            //do weights for each AnaFitParameters obj
            for(size_t j=0;j<m_fitpara.size();j++)
            {
                //cout << "FillSamples: Current par name is " << m_fitpara[j]->GetName() << endl;
                if((datatype!=0) && (((TString)(m_fitpara[j]->GetName())).Contains("par_detAve")))
                    continue;
                else if((datatype==0) && (((TString)(m_fitpara[j]->GetName())).Contains("par_detFine")))
                    continue;

                if(m_samples[s]->isIngrid()) m_fitpara[j]->ReWeightIngrid(ev, s, i, new_pars[j]);
                else m_fitpara[j]->ReWeight(ev, s, i, new_pars[j]);
            }
        }
        m_samples[s]->FillEventHisto(datatype);

        //calculate chi2 for each sample
        chi2 += m_samples[s]->CalcChi2();
        if(isGonnaBeABiggun) cout << "chi2 for sample " << s << " is " <<  m_samples[s]->CalcChi2() << endl;;
    }
    return chi2;
}

// fcn <-- this is the actual FCN
void XsecFitter::fcn(Int_t &npar, Double_t *gin, Double_t &f,
        Double_t *par, Int_t iflag)
{
    m_calls++;

    bool isGonnaBeABiggun = false;
    if((m_calls<1001 && (m_calls%100==0 || m_calls<20)) || (m_calls>1001 && m_calls%1000==0) || (m_calls>10001 && m_calls%10000==0)) isGonnaBeABiggun = true;

    if(isGonnaBeABiggun) cout << "MCHisto:" << endl;
    if(isGonnaBeABiggun) for(int i=1;i<10;i++){cout << mcHisto->GetBinContent(i) << endl;}
    if(isGonnaBeABiggun) cout << "MCSigHisto:" << endl;
    if(isGonnaBeABiggun) for(int i=1;i<10;i++){cout << mcSigHisto->GetBinContent(i) << endl;}

    //Regularisation:
    double chi2_reg=0.0;
    double chi2_reg2=0.0;

    vector< vector<double> > new_pars;
    int k=0;
    double parAvg = 0;
    double parWhtAvg = 0;
    double chi2_sys = 0.0;
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        vector<double> vec;
        for(int j=0;j<m_nparclass[i];j++){
            vec.push_back(par[k++]);
            // Regularisation beyond the template weight covariance matrix:
            //if(i==0 && j!=0) chi2_reg+= reg_p1*abs(par[k-1]-par[k-2]);
            //if(i==0 && j!=0 && j<10) chi2_reg+= reg_p1*(par[k-1]-par[k-2])*(par[k-1]-par[k-2]); // MC prior for reg
            //if(i==0 && j!=0 && j<10) chi2_reg+= reg_p1*( (mcHisto->GetBinContent(j+1)*(par[k-1]/(mcHisto->GetEntries()/8))) - (mcHisto->GetBinContent(j)*(par[k-2]/(mcHisto->GetEntries()/8))) )*( (mcHisto->GetBinContent(j+1)*(par[k-1]/(mcHisto->GetEntries()/8))) - (mcHisto->GetBinContent(j)*(par[k-2]/(mcHisto->GetEntries()/8))) ); // Flat prior for reg

            if(i==0){
                for(int p=0; p<nipsbins && j==0;p++) parAvg+=par[p];
                for(int p=0; p<nipsbins && j==0;p++) parWhtAvg+=par[p]*(mcSigHisto->GetBinContent(p+1)/mcSigHisto->Integral(0,nipsbins));
                parAvg = parAvg / nipsbins;
                if(j==0 && isGonnaBeABiggun) cout << "Fit parameter average over IPS bins is: " << parAvg << endl;
                if(j==0 && isGonnaBeABiggun) cout << "Fit parameter weighted average over IPS bins is: " << parWhtAvg << endl;
                if(j>=nipsbins) chi2_reg2 += reg_p2*(par[k-1]-parWhtAvg)*(par[k-1]-parWhtAvg); // Reg for OOPS
            }


            // End regularisation
            if(isGonnaBeABiggun) cout << "param " << j << " of class " << i << " has value " << par[k-1] << endl; // << " giving total chi2_reg " << chi2_reg << endl;
        }
        chi2_sys += m_fitpara[i]->GetChi2(vec);
        // "Systematic error" on the fit parameters comes from the regularisation covatriance matrix, store as a chi2 reg error:
        if(i==0){
            chi2_reg+=chi2_sys;
            chi2_sys -= m_fitpara[i]->GetChi2(vec);
        }

        new_pars.push_back(vec);
        if(isGonnaBeABiggun) cout << "chi2_sys contribution from param set " << i << " is " << m_fitpara[i]->GetChi2(vec) << endl;
    }
    if(isGonnaBeABiggun) cout << "chi2_sys contribution from regularisation " << chi2_reg << endl;
    //chi2_sys += chi2_reg;
    vec_chi2_sys.push_back(chi2_sys);
    vec_chi2_reg.push_back(chi2_reg);
    vec_chi2_reg2.push_back(chi2_reg2);

    double chi2_stat = FillSamples(new_pars, 0);
    vec_chi2_stat.push_back(chi2_stat);

    //save hists if requested
    if(m_calls % (m_freq*10000) == 0  && m_dir) {
        DoSaveParams(new_pars);
        if(m_calls % (m_freq*10000) == 0) {
            DoSaveEvents(m_calls);
        }
    }
    // total chi2
    f = chi2_stat + chi2_sys + chi2_reg + chi2_reg2;

    //Print status of the fit:
    if(isGonnaBeABiggun) cout << "m_calls is: " << m_calls << endl;
    if(isGonnaBeABiggun) cout << "Chi2 for this iter: " << f << endl;
    if(isGonnaBeABiggun) cout << "Chi2 stat / syst / reg1 / reg2: " << chi2_stat << " / " << chi2_sys << " / " << chi2_reg << " / " << chi2_reg2 << endl;

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
void XsecFitter::DoSaveFinalEvents(int fititer, vector< vector<double> > res_params)
{
    cout << endl << "**************************" << endl << "Fit finished, saving results .. " << endl << "**************************" << endl << endl;
    outtree= new TTree("selectedEvents", "selectedEvents");
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
                m_fitpara[j]->ReWeight(ev, s, i, res_params[j]);
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
    TH1D* histochi2reg2 = new TH1D("chi2_reg2_periter","chi2_reg2_periter", m_calls+1, 0, m_calls+1);
    TH1D* histochi2tot = new TH1D("chi2_tot_periter","chi2_tot_periter", m_calls+1, 0, m_calls+1);

    TH1D* reg_param = new TH1D("reg_param","reg_param", m_calls+1, 0, m_calls+1);
    TH1D* reg_param2 = new TH1D("reg_param2","reg_param2", m_calls+1, 0, m_calls+1);
    reg_param->Fill(reg_p1);
    reg_param2->Fill(reg_p2);

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
        histochi2reg2->SetBinContent(i+1,vec_chi2_reg2[i]);
        histochi2tot->SetBinContent(i+1,vec_chi2_sys[i] + vec_chi2_stat[i]);
    }
    m_dir->cd();
    histochi2stat->Write();
    histochi2sys->Write();
    histochi2reg->Write();
    histochi2reg2->Write();
    histochi2tot->Write();
    reg_param->Write();
    reg_param2->Write();

    delete histochi2stat;
    delete histochi2sys;
    delete histochi2reg;
    delete histochi2reg2;
    delete histochi2tot;
}

void XsecFitter::DoSaveResults(vector< vector<double> > parresults, vector< vector<double> > parerrors, vector< vector<double> > parerrorsplus, vector< vector<double> > parerrorsminus,
        vector< vector<double> > parerrorspara, vector< vector<double> > parerrorsglobc, vector< vector<double> > parerrorsprof, double chi2){
    //loop on number of parameter classes
    for(size_t i=0;i<m_fitpara.size();i++)
    {
        TH1D* histopar = new TH1D(Form("paramhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerr = new TH1D(Form("paramerrhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerrplus = new TH1D(Form("paramerrplushist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrplushist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerrminus = new TH1D(Form("paramerrminushist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrminushist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerrpara = new TH1D(Form("paramerrparahist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrparahist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerrglobc = new TH1D(Form("paramerrglobchist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrglobchist_par%s_result",m_fitpara[i]->GetName().c_str()),
                m_nparclass[i], 0, m_nparclass[i]);
        TH1D* histoparerrprof = new TH1D(Form("paramerrprofhist_par%s_result",m_fitpara[i]->GetName().c_str()),
                Form("paramerrprofhist_par%s_result",m_fitpara[i]->GetName().c_str()),
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
            histoparerrplus->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histoparerrplus->SetBinContent(j+1,(parerrorsplus[i])[j]);
            histoparerrminus->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histoparerrminus->SetBinContent(j+1,(parerrorsminus[i])[j]);
            histoparerrpara->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histoparerrpara->SetBinContent(j+1,(parerrorspara[i])[j]);
            histoparerrglobc->GetXaxis()->SetBinLabel(j+1,TString(parnames[j]));
            histoparerrprof->SetBinContent(j+1,(parerrorsprof[i])[j]);
        }
        m_dir->cd();
        histopar->Write();
        histoparerr->Write();
        histoparerrplus->Write();
        histoparerrminus->Write();
        histoparerrpara->Write();
        histoparerrglobc->Write();
        histoparerrprof->Write();
        delete histopar;
        delete histoparerr;
        delete histoparerrplus;
        delete histoparerrminus;
        delete histoparerrpara;
        delete histoparerrglobc;
        delete histoparerrprof;
    }
    double x[1]={0};
    double y[1]={chi2};
    TGraph* grapchi2 = new TGraph(1,x,y);
    grapchi2->SetName("final_chi2_fromMinuit");
    m_dir->cd();
    grapchi2->Write();

    std::vector<std::string> topology = {"cc0pi0p", "cc0pi1p", "cc0pinp", "cc1pi+", "ccother",
        "backg", "Null", "OOFV"};

    //ADDED save actual final results
    for(size_t s=0;s<m_samples.size();s++)
    {
        m_samples[s]->GetSampleBreakdown(m_dir, "fit", topology, true);
    }
}

void XsecFitter::CollectSampleHistos(){
    //loop over samples
    for(size_t s=0;s<m_samples.size();s++)
    {
        AnySample* sam = ((AnySample*)m_samples[s]);
        if(s==0){
            mcHisto = (TH1D*)((sam->GetMCTruthHisto())->Clone("mcHisto"));
            mcSigHisto = (TH1D*)((sam->GetSignalHisto())->Clone("mcSigHisto"));
        }
        else{
            mcHisto->Add((sam->GetMCTruthHisto()));
            mcSigHisto->Add((sam->GetSignalHisto()));
        }
        //cout << "MCHisto after sample :" << s << endl;
        //for(int i=1;i<10;i++){cout << mcHisto->GetBinContent(i) << endl;}
    }
}

double XsecFitter::FindProfiledError(int param, TMatrixDSym mat){
    double error = 0.0;
    double maxerror = 0.0;
    TMatrixDSymEigen me(mat);

    TVectorD eigenval = me.GetEigenValues();
    TMatrixD eigenvec = me.GetEigenVectors();

    for(int i=0; i<mat.GetNrows(); i++){
        error = fabs((eigenval[i])*(eigenvec[param][i]));
        if(error>maxerror) maxerror=error;
    }

    return sqrt(maxerror);
}

