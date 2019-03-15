#include "XsecFitter.hh"

XsecFitter::XsecFitter(TDirectory* dirout, const int seed, const int num_threads)
    : rng(new TRandom3(seed))
    , m_fitter(nullptr)
    , m_fcn(nullptr)
    , m_dir(dirout)
    , m_save(false)
    , m_zerosyst(false)
    , m_freq(10000)
    , m_threads(num_threads)
    , m_potratio(1.0)
    , m_npar(0)
    , m_calls(0)
{
    gRandom = rng;

    min_settings.minimizer = "Minuit2";
    min_settings.algorithm = "Migrad";
    min_settings.print_level = 2;
    min_settings.strategy  = 1;
    min_settings.tolerance = 1E-2;
    min_settings.max_iter  = 1E6;
    min_settings.max_fcn   = 1E9;
}

XsecFitter::XsecFitter(TDirectory* dirout, const int seed)
    : XsecFitter(dirout, seed, 1)
{
}

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

void XsecFitter::SetSeed(int seed)
{
    if(rng == nullptr)
    {
        rng     = new TRandom3(seed);
        gRandom = rng; // Global pointer
    }
    else
        rng->SetSeed(seed);
}

void XsecFitter::FixParameter(const std::string& par_name, const double& value)
{
    auto iter = std::find(par_names.begin(), par_names.end(), par_name);
    if(iter != par_names.end())
    {
        const int i = std::distance(par_names.begin(), iter);
        m_fitter->SetVariable(i, par_names.at(i).c_str(), value, 0);
        m_fitter->FixVariable(i);
        std::cout << TAG << "Fixing parameter " << par_names.at(i) << " to value " << value
                  << std::endl;
    }
    else
    {
        std::cerr << ERR << "In function XsecFitter::FixParameter()\n"
                  << ERR << "Parameter " << par_name << " not found!" << std::endl;
    }
}

void XsecFitter::SetMinSettings(const MinSettings& ms)
{
    min_settings = ms;
    if(m_fitter != nullptr)
    {
        m_fitter->SetStrategy(min_settings.strategy);
        m_fitter->SetPrintLevel(min_settings.print_level);
        m_fitter->SetTolerance(min_settings.tolerance);
        m_fitter->SetMaxIterations(min_settings.max_iter);
        m_fitter->SetMaxFunctionCalls(min_settings.max_fcn);
    }
}

void XsecFitter::InitFitter(std::vector<AnaFitParameters*>& fitpara)
{
    m_fitpara = fitpara;
    std::vector<double> par_step, par_low, par_high;
    std::vector<bool> par_fixed;

    for(std::size_t i = 0; i < m_fitpara.size(); i++)
    {
        m_npar += m_fitpara[i]->GetNpar();

        std::vector<std::string> vec0;
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

        std::vector<bool> vec3;
        m_fitpara[i]->GetParFixed(vec3);
        par_fixed.insert(par_fixed.end(), vec3.begin(), vec3.end());
    }

    if(m_npar == 0)
    {
        std::cerr << ERR << "No fit parameters were defined." << std::endl;
        return;
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "           Initilizing fitter              " << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << TAG << "Minimizer settings..." << std::endl
              << TAG << "Minimizer: " << min_settings.minimizer << std::endl
              << TAG << "Algorithm: " << min_settings.algorithm << std::endl
              << TAG << "Strategy : " << min_settings.strategy << std::endl
              << TAG << "Print Lvl: " << min_settings.print_level << std::endl
              << TAG << "Tolerance: " << min_settings.tolerance << std::endl
              << TAG << "Max Iterations: " << min_settings.max_iter << std::endl
              << TAG << "Max Fcn Calls : " << min_settings.max_fcn << std::endl;

    m_fitter = ROOT::Math::Factory::CreateMinimizer(min_settings.minimizer.c_str(), min_settings.algorithm.c_str());
    m_fcn    = new ROOT::Math::Functor(this, &XsecFitter::CalcLikelihood, m_npar);

    m_fitter->SetFunction(*m_fcn);
    m_fitter->SetStrategy(min_settings.strategy);
    m_fitter->SetPrintLevel(min_settings.print_level);
    m_fitter->SetTolerance(min_settings.tolerance);
    m_fitter->SetMaxIterations(min_settings.max_iter);
    m_fitter->SetMaxFunctionCalls(min_settings.max_fcn);

    for(int i = 0; i < m_npar; ++i)
    {
        m_fitter->SetVariable(i, par_names[i], par_prefit[i], par_step[i]);
        m_fitter->SetVariableLimits(i, par_low[i], par_high[i]);

        if(par_fixed[i] == true)
            m_fitter->FixVariable(i);
    }

    std::cout << TAG << "Number of defined parameters: " << m_fitter->NDim() << std::endl
              << TAG << "Number of free parameters   : " << m_fitter->NFree() << std::endl
              << TAG << "Number of fixed parameters  : " << m_fitter->NDim() - m_fitter->NFree()
              << std::endl;

    TH1D h_prefit("hist_prefit_par_all", "hist_prefit_par_all", m_npar, 0, m_npar);
    int num_par = 1;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        for(int j = 0; j < m_fitpara[i]->GetNpar(); ++j)
        {
            h_prefit.SetBinContent(num_par, m_fitpara[i]->GetParPrior(j));
            if(m_fitpara[i]->HasCovMat())
            {
                TMatrixDSym* covMat = m_fitpara[i]->GetCovMat();
                h_prefit.SetBinError(num_par, std::sqrt((*covMat)[j][j]));
            }
            else
                h_prefit.SetBinError(num_par, 0);
            num_par++;
        }
    }
    m_dir->cd();
    h_prefit.Write();
}

bool XsecFitter::Fit(const std::vector<AnaSample*>& samples, int fit_type, bool stat_fluc)
{
    std::cout << TAG << "Starting to fit." << std::endl;
    m_samples = samples;

    if(m_fitter == nullptr)
    {
        std::cerr << ERR << "In XsecFitter::Fit()\n"
                  << ERR << "Fitter has not been initialized." << std::endl;
        return false;
    }

    if(fit_type == kAsimovFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kAsimov, stat_fluc);
    }
    else if(fit_type == kExternalFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kExternal, stat_fluc);
    }
    else if(fit_type == kDataFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kData, stat_fluc);
    }
    else if(fit_type == kToyFit)
    {
        GenerateToyData(0, stat_fluc);
    }
    else
    {
        std::cerr << ERR << "In XsecFitter::Fit()\n"
                  << ERR << "No valid fitting mode provided." << std::endl;
        return false;
    }

    SaveEvents(m_calls);

    bool did_converge = false;
    std::cout << TAG << "Fit prepared." << std::endl;
    std::cout << TAG << "Calling Minimize, running " << min_settings.algorithm << std::endl;
    did_converge = m_fitter->Minimize();

    if(!did_converge)
    {
        std::cout << ERR << "Fit did not converge while running " << min_settings.algorithm
                  << std::endl;
        std::cout << ERR << "Failed with status code: " << m_fitter->Status() << std::endl;
    }
    else
    {
        std::cout << TAG << "Fit converged." << std::endl
                  << TAG << "Status code: " << m_fitter->Status() << std::endl;

        std::cout << TAG << "Calling HESSE." << std::endl;
        did_converge = m_fitter->Hesse();
    }

    if(!did_converge)
    {
        std::cout << ERR << "Hesse did not converge." << std::endl;
        std::cout << ERR << "Failed with status code: " << m_fitter->Status() << std::endl;
    }
    else
    {
        std::cout << TAG << "Hesse converged." << std::endl
                  << TAG << "Status code: " << m_fitter->Status() << std::endl;
    }

    if(m_dir)
        SaveChi2();

    const int ndim        = m_fitter->NDim();
    const int nfree       = m_fitter->NFree();
    const double* par_val = m_fitter->X();
    const double* par_err = m_fitter->Errors();
    double cov_array[ndim * ndim];
    m_fitter->GetCovMatrix(cov_array);

    std::vector<double> par_val_vec(par_val, par_val + ndim);
    std::vector<double> par_err_vec(par_err, par_err + ndim);

    unsigned int par_offset = 0;
    TMatrixDSym cov_matrix(ndim, cov_array);
    for(const auto& fit_param : m_fitpara)
    {
        if(fit_param->IsDecomposed())
        {
            cov_matrix  = fit_param->GetOriginalCovMat(cov_matrix, par_offset);
            par_val_vec = fit_param->GetOriginalParameters(par_val_vec, par_offset);
        }
        par_offset += fit_param->GetNpar();
    }

    TMatrixDSym cor_matrix(ndim);
    for(int r = 0; r < ndim; ++r)
    {
        for(int c = 0; c < ndim; ++c)
        {
            cor_matrix[r][c] = cov_matrix[r][c] / std::sqrt(cov_matrix[r][r] * cov_matrix[c][c]);
            if(std::isnan(cor_matrix[r][c]))
                cor_matrix[r][c] = 0;
        }
    }

    TVectorD postfit_param(ndim, &par_val_vec[0]);
    std::vector<std::vector<double>> res_pars;
    std::vector<std::vector<double>> err_pars;
    int k = 0;
    for(int i = 0; i < m_fitpara.size(); i++)
    {
        const unsigned int npar = m_fitpara[i]->GetNpar();
        std::vector<double> vec_res;
        std::vector<double> vec_err;

        for(int j = 0; j < npar; j++)
        {
            vec_res.push_back(par_val_vec[k]);
            vec_err.push_back(std::sqrt(cov_matrix[k][k]));
            k++;
        }

        res_pars.emplace_back(vec_res);
        err_pars.emplace_back(vec_err);
    }

    if(k != ndim)
    {
        std::cout << ERR << "Number of parameters does not match." << std::endl;
        return false;
    }

    m_dir->cd();
    cov_matrix.Write("res_cov_matrix");
    cor_matrix.Write("res_cor_matrix");
    postfit_param.Write("res_vector");

    SaveResults(res_pars, err_pars);
    SaveFinalEvents(m_calls, res_pars);

    if(!did_converge)
        std::cout << ERR << "Not valid fit result." << std::endl;
    std::cout << TAG << "Fit routine finished. Results saved." << std::endl;

    return did_converge;
}

void XsecFitter::GenerateToyData(int toy_type, bool stat_fluc)
{
    double chi2_stat = 0.0;
    double chi2_syst = 0.0;
    std::vector<std::vector<double>> fitpar_throw;
    for(const auto& fitpar : m_fitpara)
    {
        std::vector<double> toy_throw(fitpar->GetNpar(), 0.0);
        fitpar -> ThrowPar(toy_throw);

        chi2_syst += fitpar -> GetChi2(toy_throw);
        fitpar_throw.emplace_back(toy_throw);
    }

    for(int s = 0; s < m_samples.size(); ++s)
    {
        const unsigned int N  = m_samples[s]->GetN();
        const std::string det = m_samples[s]->GetDetector();
#pragma omp parallel for num_threads(m_threads)
        for(unsigned int i = 0; i < N; ++i)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->ResetEvWght();
            for(int j = 0; j < m_fitpara.size(); ++j)
                m_fitpara[j]->ReWeight(ev, det, s, i, fitpar_throw[j]);
        }

        m_samples[s]->FillEventHist(kAsimov, stat_fluc);
        m_samples[s]->FillEventHist(kReset);
        chi2_stat += m_samples[s]->CalcChi2();
    }

    std::cout << TAG << "Generated toy throw from parameters.\n"
              << TAG << "Initial Chi2 Syst: " << chi2_syst << std::endl
              << TAG << "Initial Chi2 Stat: " << chi2_stat << std::endl;

    SaveParams(fitpar_throw);
}

double XsecFitter::FillSamples(std::vector<std::vector<double>>& new_pars, int datatype)
{
    double chi2      = 0.0;
    bool output_chi2 = false;
    if((m_calls < 1001 && (m_calls % 100 == 0 || m_calls < 20))
       || (m_calls > 1001 && m_calls % 1000 == 0))
        output_chi2 = true;

    unsigned int par_offset = 0;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        if(m_fitpara[i]->IsDecomposed())
        {
            new_pars[i] = m_fitpara[i]->GetOriginalParameters(new_pars[i]);
        }
        par_offset += m_fitpara[i]->GetNpar();
    }

    for(int s = 0; s < m_samples.size(); ++s)
    {
        const unsigned int num_events = m_samples[s]->GetN();
        const std::string det         = m_samples[s]->GetDetector();
#pragma omp parallel for num_threads(m_threads)
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->ResetEvWght();
            for(int j = 0; j < m_fitpara.size(); ++j)
            {
                m_fitpara[j]->ReWeight(ev, det, s, i, new_pars[j]);
            }
        }

        m_samples[s]->FillEventHist(datatype);
        double sample_chi2 = m_samples[s]->CalcChi2();
        chi2 += sample_chi2;

        if(output_chi2)
        {
            std::cout << TAG << "Chi2 for sample " << m_samples[s]->GetName() << " is "
                      << sample_chi2 << std::endl;
        }
    }

    return chi2;
}

double XsecFitter::CalcLikelihood(const double* par)
{
    m_calls++;

    bool output_chi2 = false;
    if((m_calls < 1001 && (m_calls % 100 == 0 || m_calls < 20))
       || (m_calls > 1001 && m_calls % 1000 == 0))
        output_chi2 = true;

    int k           = 0;
    double chi2_sys = 0.0;
    double chi2_reg = 0.0;
    std::vector<std::vector<double>> new_pars;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        const unsigned int npar = m_fitpara[i]->GetNpar();
        std::vector<double> vec;
        for(int j = 0; j < npar; ++j)
        {
            //if(output_chi2)
            //    std::cout << "Parameter " << j << " for " << m_fitpara[i]->GetName()
            //              << " has value " << par[k] << std::endl;
            vec.push_back(par[k++]);
        }

        if(!m_zerosyst)
            chi2_sys += m_fitpara[i]->GetChi2(vec);

        if(m_fitpara[i]->IsRegularised())
            chi2_reg += m_fitpara[i]->CalcRegularisation(vec);

        new_pars.push_back(vec);
        if(output_chi2)
        {
            std::cout << TAG << "Chi2 contribution from " << m_fitpara[i]->GetName() << " is "
                      << m_fitpara[i]->GetChi2(vec) << std::endl;
        }
    }

    double chi2_stat = FillSamples(new_pars, kMC);
    vec_chi2_stat.push_back(chi2_stat);
    vec_chi2_sys.push_back(chi2_sys);
    vec_chi2_reg.push_back(chi2_reg);

    if(m_calls % m_freq == 0 && m_save)
    {
        SaveParams(new_pars);
        SaveEvents(m_calls);
    }

    if(output_chi2)
    {
        std::cout << TAG << "Func Calls: " << m_calls << std::endl;
        std::cout << TAG << "Chi2 total: " << chi2_stat + chi2_sys + chi2_reg << std::endl;
        std::cout << TAG << "Chi2 stat : " << chi2_stat << std::endl
                  << TAG << "Chi2 syst : " << chi2_sys  << std::endl
                  << TAG << "Chi2 reg  : " << chi2_reg  << std::endl;
    }

    return chi2_stat + chi2_sys + chi2_reg;
}

void XsecFitter::SaveEvents(int fititer)
{
    for(size_t s = 0; s < m_samples.size(); s++)
    {
        m_samples[s]->Write(m_dir, Form("evhist_sam%d_iter%d", (int)s, m_calls), fititer);
    }
}

void XsecFitter::SaveFinalEvents(int fititer, std::vector<std::vector<double>>& res_params)
{
    outtree = new TTree("selectedEvents", "selectedEvents");
    InitOutputTree();
    for(size_t s = 0; s < m_samples.size(); s++)
    {
        m_samples[s]->Write(m_dir, Form("evhist_sam%d_finaliter", (int)s), fititer);
        for(int i = 0; i < m_samples[s]->GetN(); i++)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            for(size_t j = 0; j < m_fitpara.size(); j++)
            {
                std::string det = m_samples[s]->GetDetector();
                m_fitpara[j]->ReWeight(ev, det, s, i, res_params[j]);
            }

            sample    = ev->GetSampleType();
            D1true    = ev->GetTrueD1();
            D2true    = ev->GetTrueD2();
            topology  = ev->GetTopology();
            reaction  = ev->GetReaction();
            target    = ev->GetTarget();
            nutype    = ev->GetFlavor();
            D1Reco    = ev->GetRecoD1();
            D2Reco    = ev->GetRecoD2();
            weightNom = (ev->GetEvWght()) * (ev->GetEvWghtMC());
            weightMC  = ev->GetEvWghtMC();
            weight    = ev->GetEvWght() * m_potratio;
            outtree->Fill();
        }
    }
    m_dir->cd();
    outtree->Write();
}

void XsecFitter::SaveParams(const std::vector<std::vector<double>>& new_pars)
{
    std::vector<double> temp_vec;
    for(size_t i = 0; i < m_fitpara.size(); i++)
    {
        const unsigned int npar = m_fitpara[i]->GetNpar();
        const std::string name  = m_fitpara[i]->GetName();
        std::stringstream ss;

        ss << "hist_" << name << "_iter" << m_calls;
        TH1D h_par(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        std::vector<std::string> vec_names;
        m_fitpara[i]->GetParNames(vec_names);
        for(int j = 0; j < npar; j++)
        {
            h_par.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par.SetBinContent(j + 1, new_pars[i][j]);
            temp_vec.emplace_back(new_pars[i][j]);
        }
        m_dir->cd();
        h_par.Write();
    }

    TVectorD root_vec(temp_vec.size(), &temp_vec[0]);
    root_vec.Write(Form("vec_par_all_iter%d", m_calls));
}

void XsecFitter::SaveChi2()
{
    TH1D h_chi2stat("chi2_stat_periter", "chi2_stat_periter", m_calls + 1, 0, m_calls + 1);
    TH1D h_chi2sys("chi2_sys_periter", "chi2_sys_periter", m_calls + 1, 0, m_calls + 1);
    TH1D h_chi2reg("chi2_reg_periter", "chi2_reg_periter", m_calls + 1, 0, m_calls + 1);
    TH1D h_chi2tot("chi2_tot_periter", "chi2_tot_periter", m_calls + 1, 0, m_calls + 1);

    if(vec_chi2_stat.size() != vec_chi2_sys.size())
    {
        std::cout << ERR << "Number of saved iterations for chi2 stat and chi2 syst are different."
                  << std::endl;
    }
    for(size_t i = 0; i < vec_chi2_stat.size(); i++)
    {
        h_chi2stat.SetBinContent(i + 1, vec_chi2_stat[i]);
        h_chi2sys.SetBinContent(i + 1, vec_chi2_sys[i]);
        h_chi2reg.SetBinContent(i + 1, vec_chi2_reg[i]);
        h_chi2tot.SetBinContent(i + 1, vec_chi2_sys[i] + vec_chi2_stat[i] + vec_chi2_reg[i]);
    }

    m_dir->cd();
    h_chi2stat.Write();
    h_chi2sys.Write();
    h_chi2reg.Write();
    h_chi2tot.Write();
}

void XsecFitter::SaveResults(const std::vector<std::vector<double>>& par_results,
                             const std::vector<std::vector<double>>& par_errors)
{
    for(std::size_t i = 0; i < m_fitpara.size(); i++)
    {
        const unsigned int npar = m_fitpara[i]->GetNpar();
        const std::string name  = m_fitpara[i]->GetName();
        std::vector<double> par_original;
        m_fitpara[i]->GetParOriginal(par_original);

        TMatrixDSym* cov_mat = m_fitpara[i]->GetOriginalCovMat();

        std::stringstream ss;

        ss << "hist_" << name << "_result";
        TH1D h_par_final(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_prior";
        TH1D h_par_prior(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_error_final";
        TH1D h_err_final(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        ss.str("");
        ss << "hist_" << name << "_error_prior";
        TH1D h_err_prior(ss.str().c_str(), ss.str().c_str(), npar, 0, npar);

        std::vector<std::string> vec_names;
        m_fitpara[i]->GetParNames(vec_names);
        for(int j = 0; j < npar; j++)
        {
            h_par_final.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par_final.SetBinContent(j + 1, par_results[i][j]);
            h_par_prior.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_par_prior.SetBinContent(j + 1, par_original[j]);
            h_err_final.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_err_final.SetBinContent(j + 1, par_errors[i][j]);

            double err_prior = 0.0;
            if(cov_mat != nullptr)
                err_prior = TMath::Sqrt((*cov_mat)(j,j));

            h_err_prior.GetXaxis()->SetBinLabel(j + 1, vec_names[j].c_str());
            h_err_prior.SetBinContent(j + 1, err_prior);
        }

        m_dir->cd();
        h_par_final.Write();
        h_par_prior.Write();
        h_err_final.Write();
        h_err_prior.Write();
    }
}
