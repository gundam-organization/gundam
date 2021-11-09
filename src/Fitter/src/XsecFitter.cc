#include "XsecFitter.hh"

XsecFitter::XsecFitter(TDirectory* dirout, const int seed, const int num_threads)
    : rng(new TRandom3(seed))
    , m_fitter(nullptr)
    , m_fcn(nullptr)
    , m_dir(dirout)
    , m_save(false)
    , m_save_events(true)
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

// Initializes the fit by setting up the fit parameters and creating the ROOT minimizer:
void XsecFitter::InitFitter(std::vector<AnaFitParameters*>& fitpara)
{
    // Vector of the different parameter types such as [template, flux, detector, cross section]:
    m_fitpara = fitpara;

    // Vectors holding the settings for the fit parameters (for all parameter types):
    std::vector<double> par_step, par_low, par_high;
    std::vector<bool> par_fixed;

    // ROOT random number interface (seed of 0 means that seed is automatically computed based on time):
    //TRandom3 rng(0);

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(std::size_t i = 0; i < m_fitpara.size(); i++)
    {
        // m_npar is the number of total fit paramters:
        m_npar += m_fitpara[i]->GetNpar();

        // Get names of all the different parameters (for all parameter types) and store them in par_names:
        std::vector<std::string> vec0;
        m_fitpara[i]->GetParNames(vec0);
        par_names.insert(par_names.end(), vec0.begin(), vec0.end());

        // Get the priors for this parameter type (should be 1 unless decomp has been set to true in the .json config file) and store them in vec1:
        std::vector<double> vec1, vec2;
        m_fitpara[i]->GetParPriors(vec1);

        // If rng_template has been set to true in the .json config file, the template parameters will be randomized (around 1) according to a gaussian distribution:
        if(m_fitpara[i]->DoRNGstart())
        {
            std::cout << TAG << "Randomizing start point for " << m_fitpara[i]->GetName() << std::endl;
            for(auto& p : vec1)
                p += (p * rng->Gaus(0.0, 0.1));
        }

        // Store the prefit values (for all parameter types) in par_prefit:
        par_prefit.insert(par_prefit.end(), vec1.begin(), vec1.end());

        // Store the pars_step values (for all parameter types) and store them in par_step:
        m_fitpara[i]->GetParSteps(vec1);
        par_step.insert(par_step.end(), vec1.begin(), vec1.end());

        // Store the lower and upper limits for the fit parameters (for all parameter types) in par_low and par_high:
        m_fitpara[i]->GetParLimits(vec1, vec2);
        par_low.insert(par_low.end(), vec1.begin(), vec1.end());
        par_high.insert(par_high.end(), vec2.begin(), vec2.end());

        // Store the flags indicating whether a parameter is fixed (for all parameter types) in par_fixed:
        std::vector<bool> vec3;
        m_fitpara[i]->GetParFixed(vec3);
        par_fixed.insert(par_fixed.end(), vec3.begin(), vec3.end());
    }

    // Nothing to fit with zero fit parameters:
    if(m_npar == 0)
    {
        std::cerr << ERR << "No fit parameters were defined." << std::endl;
        return;
    }

    // Print information about the minimizer settings specified in the .json config file:
    std::cout << "===========================================" << std::endl;
    std::cout << "           Initializing fitter             " << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << TAG << "Minimizer settings..." << std::endl
              << TAG << "Minimizer: " << min_settings.minimizer << std::endl
              << TAG << "Algorithm: " << min_settings.algorithm << std::endl
              << TAG << "Likelihood: " << min_settings.likelihood << std::endl
              << TAG << "Strategy : " << min_settings.strategy << std::endl
              << TAG << "Print Lvl: " << min_settings.print_level << std::endl
              << TAG << "Tolerance: " << min_settings.tolerance << std::endl
              << TAG << "Max Iterations: " << min_settings.max_iter << std::endl
              << TAG << "Max Fcn Calls : " << min_settings.max_fcn << std::endl;

    // Create ROOT minimizer of given minimizerType and algoType:
    m_fitter = ROOT::Math::Factory::CreateMinimizer(min_settings.minimizer.c_str(), min_settings.algorithm.c_str());

    // The ROOT Functor class is used to wrap multi-dimensional function objects, in this case the XsecFitter::evalFit function calculates and returns chi2_stat + chi2_sys + chi2_reg in each iteration of the fitter:
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
        //m_fitter->SetVariableLimits(i, par_low[i], par_high[i]);

        if(par_fixed[i] == true)
            m_fitter->FixVariable(i);
    }

    std::cout << TAG << "Number of defined parameters: " << m_fitter->NDim() << std::endl
              << TAG << "Number of free parameters   : " << m_fitter->NFree() << std::endl
              << TAG << "Number of fixed parameters  : " << m_fitter->NDim() - m_fitter->NFree()
              << std::endl;

    TH1D h_prefit("hist_prefit_par_all", "hist_prefit_par_all", m_npar, 0, m_npar);
    TVectorD v_prefit_original(m_npar);
    TVectorD v_prefit_decomp(m_npar);
    TVectorD v_prefit_start(m_npar, par_prefit.data());

    int num_par = 1;
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        TMatrixDSym* cov_mat = m_fitpara[i]->GetCovMat();
        for(int j = 0; j < m_fitpara[i]->GetNpar(); ++j)
        {
            h_prefit.SetBinContent(num_par, m_fitpara[i]->GetParPrior(j));
            if(m_fitpara[i]->HasCovMat())
                h_prefit.SetBinError(num_par, std::sqrt((*cov_mat)[j][j]));
            else
                h_prefit.SetBinError(num_par, 0);

            v_prefit_original[num_par-1] = m_fitpara[i]->GetParOriginal(j);
            v_prefit_decomp[num_par-1] = m_fitpara[i]->GetParPrior(j);
            num_par++;
        }
    }

    m_dir->cd();
    h_prefit.Write();
    v_prefit_original.Write("vec_prefit_original");
    v_prefit_decomp.Write("vec_prefit_decomp");
    v_prefit_start.Write("vec_prefit_start");
}

bool XsecFitter::Fit(const std::vector<AnaSample*>& samples, int fit_type, bool stat_fluc)
{
    std::cout << TAG << "Starting to fit." << std::endl;

    // Vector of AnaSample objects:
    m_samples = samples;

    // Fitter should have been initialized by now with InitFitter():
    if(m_fitter == nullptr)
    {
        std::cerr << ERR << "In XsecFitter::Fit()\n"
                  << ERR << "Fitter has not been initialized." << std::endl;
        return false;
    }

    // Check what fit-type was specified in the .json config file:

    // fit-type = kAsimovFit = 0:
    if(fit_type == kAsimovFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kAsimov, stat_fluc);
    }

    // fit-type = kAsimovFit = 1:
    else if(fit_type == kExternalFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kExternal, stat_fluc);
    }

    // fit-type = kAsimovFit = 2:
    else if(fit_type == kDataFit)
    {
        for(std::size_t s = 0; s < m_samples.size(); s++)
            m_samples[s]->FillEventHist(kData, stat_fluc);
    }

    // fit-type = kAsimovFit = 3:
    else if(fit_type == kToyFit)
    {
        GenerateToyData(0, stat_fluc);
    }

    // Exit if no valid fit-type specified:
    else
    {
        std::cerr << ERR << "In XsecFitter::Fit()\n"
                  << ERR << "No valid fitting mode provided." << std::endl;
        return false;
    }

    SaveEventHist(m_calls);

    // did_converge flag which is returned at the end:
    bool did_converge = false;

    // Info messages:
    std::cout << TAG << "Fit prepared." << std::endl;
    std::cout << TAG << "Calling Minimize, running " << min_settings.algorithm << std::endl;

    // Run the actual fitter:
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

    TVectorD postfit_globalcc(ndim);
    for(int i = 0; i < ndim; ++i)
        postfit_globalcc[i] = m_fitter->GlobalCC(i);

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
    postfit_globalcc.Write("res_globalcc");

    SaveResults(res_pars, err_pars);
    SaveEventHist(m_calls, true);

    if(m_save_events)
        SaveEventTree(res_pars);

    if(!did_converge)
        std::cout << ERR << "Not valid fit result." << std::endl;
    std::cout << TAG << "Fit routine finished. Results saved." << std::endl;

    return did_converge;
}

void XsecFitter::GenerateToyData(int toy_type, bool stat_fluc)
{
    int temp_seed = rng->GetSeed();
    double chi2_stat = 0.0;
    double chi2_syst = 0.0;
    std::vector<std::vector<double>> fitpar_throw;
    for(const auto& fitpar : m_fitpara)
    {
        std::vector<double> toy_throw(fitpar->GetNpar(), 0.0);
        fitpar -> ThrowPar(toy_throw, temp_seed++);

        chi2_syst += fitpar -> GetChi2(toy_throw);
        fitpar_throw.emplace_back(toy_throw);
    }

    for(int s = 0; s < m_samples.size(); ++s)
    {
        const unsigned int N  = m_samples[s]->GetN();
        const std::string det = m_samples[s]->GetDetector();
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_threads)
#endif
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

// Loops over all samples and all bins therein, then resets event weights based on current fit parameter values, updates m_hpred, m_hmc, m_hmc_true and m_hsig histograms accordingly and computes the chi2 value which is returned:
double XsecFitter::FillSamples(std::vector<std::vector<double>>& new_pars, int datatype)
{
    // generateFormula chi2 variable which will be updated below and then returned:
    double chi2      = 0.0;

    // If the output_chi2 flag is true the chi2 contributions from the different samples are printed:
    bool output_chi2 = false;

    // Print chi2 contributions for the first 19 function calls then for every 100th function call for the first 1000 function calls and then every 1000th function call:
    if((m_calls < 1001 && (m_calls % 100 == 0 || m_calls < 20))
       || (m_calls > 1001 && m_calls % 1000 == 0))
        output_chi2 = true;

    // par_offset stores the number of fit parameters for all parameter types:
    unsigned int par_offset = 0;

    // Loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        // If we performed an eigendecomposition for this parameter type, we change the eigendecomposed input parameters back to the original parameters:
        if(m_fitpara[i]->IsDecomposed())
        {
            new_pars[i] = m_fitpara[i]->GetOriginalParameters(new_pars[i]);
        }

        // Update number of fit parameters as we loop through the different parameter types:
        par_offset += m_fitpara[i]->GetNpar();
    }

    // Loop over the different selection samples defined in the .json config file:
    for(int s = 0; s < m_samples.size(); ++s)
    {
        // Get number of events within the current sample:
        const unsigned int num_events = m_samples[s]->GetN();

        // Get the name of the detector for the current sample (as defined in the .json config file):
        const std::string det         = m_samples[s]->GetDetector();

        // Loop over all events in the current sample (this loop will be divided amongst the different __nb_threads__):
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_threads)
#endif
        for(unsigned int i = 0; i < num_events; ++i)
        {
            // Get ith event (which contains the event information such as topology, reaction, truth/reco variables, event weights, etc.):
            AnaEvent* ev = m_samples[s]->GetEvent(i);

            // reset the event weight to the original one from Highland:
            ev->ResetEvWght();

            // Loop over all the different parameter types such as [template, flux, detector, cross section]:
            for(int j = 0; j < m_fitpara.size(); ++j)
            {
//                std::cout << "Reweighting for "<< m_fitpara[j]->GetName() << std::endl;
                // Multiply the current event weight for event ev with the paramter of the current parameter type for the (truth bin)/(reco bin)/(energy bin) that this event falls in:
                m_fitpara[j]->ReWeight(ev, det, s, i, new_pars[j]);
            }
        }

        // reset m_hpred, m_hmc, m_hmc_true and m_hsig and then fill them with the updated events:
        m_samples[s]->FillEventHist(datatype);

        // Compute chi2 for the current sample (done with AnaSample::CalcLLH):
        //double sample_chi2 = m_samples[s]->CalcChi2();
        double sample_chi2 = m_samples[s]->CalcLLH();
        //double sample_chi2 = m_samples[s]->CalcEffLLH();

        // Add the chi2 contribution from the current sample to the total chi2 variable:
        chi2 += sample_chi2;

        // If output_chi2 has been set to true before, the chi2 contribution from this sample is printed:
        if(output_chi2)
        {
            std::cout << TAG << "Chi2 for sample " << m_samples[s]->GetName() << " is "
                      << sample_chi2 << std::endl;
        }
    }

    // The total chi2 value is returned:
    return chi2;
}

// Function which is called in each iteration of the fitter to calculate and return chi2_stat + chi2_sys + chi2_reg:
double XsecFitter::CalcLikelihood(const double* par)
{
    // Increase the number of function calls by 1:
    m_calls++;

    // If the output_chi2 flag is true the chi2 contributions from the different parameter types such as [template, flux, detector, cross section] are printed:
    bool output_chi2 = false;

    // Print chi2 contributions for the first 19 function calls then for every 100th function call for the first 1000 function calls and then every 1000th function call:
    if((m_calls < 1001 && (m_calls % 100 == 0 || m_calls < 20))
       || (m_calls > 1001 && m_calls % 1000 == 0))
        output_chi2 = true;

    // Index for looping over fit parameters of this parameter type:
    int k           = 0;

    // Penalty terms for systematic uncertainties and regularization:
    double chi2_sys = 0.0;
    double chi2_reg = 0.0;

    // Vector to store all the parameter values of all parameter types:
    std::vector<std::vector<double>> new_pars;

    // loop over all the different parameter types such as [template, flux, detector, cross section]:
    for(int i = 0; i < m_fitpara.size(); ++i)
    {
        // Number of fit parameters for this parameter type:
        const unsigned int npar = m_fitpara[i]->GetNpar();

        // vec stores the parameter values of this iteration for this parameter type:
        std::vector<double> vec;

        // Loop over fit parameters for this parameter type:
        for(int j = 0; j < npar; ++j)
        {
//            if(output_chi2)
//                std::cout << "Parameter " << j << " for " << m_fitpara[i]->GetName()
//                          << " has value " << par[k] << std::endl;
            
            // Fill vec with the parameter values of this iteration for this parameter type:
            vec.push_back(par[k++]);
        }

        // If we are not using zero systematics, the systematic chi2 value is computed with AnaFitParameters::GetChi2 for this parameter type and added to chi2_sys:
        if(!m_zerosyst)
            chi2_sys += m_fitpara[i]->GetChi2(vec);

        // If we are using regularization, the regularization chi2 value is computed with FitParameters::CalcRegularisation for this parameter type and added to chi2_reg:
        if(m_fitpara[i]->IsRegularised())
            chi2_reg += m_fitpara[i]->CalcRegularisation(vec);

        // Fill new_pars with the parameter values of all parameter types:
        new_pars.push_back(vec);

        // If output_chi2 has been set to true before, the chi2 contribution from this parameter type is printed:
        if(output_chi2)
        {
            std::cout << TAG << "Chi2 contribution from " << m_fitpara[i]->GetName() << " is "
                      << m_fitpara[i]->GetChi2(vec) << std::endl;
        }
    }

    // reset event weights based on current fit parameter values, update m_hpred, m_hmc, m_hmc_true and m_hsig histograms accordingly and compute the chi2_stat value:
    double chi2_stat = FillSamples(new_pars, kMC);

    // The different chi2 values for the current iteration of the fitter are stored in the corresponding vectors:
    vec_chi2_stat.push_back(chi2_stat);
    vec_chi2_sys.push_back(chi2_sys);
    vec_chi2_reg.push_back(chi2_reg);

    // If the m_save flag has been set to true, the fit parameters for the current iteration are saved with the given frequency m_freq:
    if(m_calls % m_freq == 0 && m_save)
    {
        SaveParams(new_pars);
        SaveEventHist(m_calls);
    }

    // If output_chi2 has been set to true before, the different chi2 contributions are printed:
    if(output_chi2)
    {
        std::cout << TAG << "Func Calls: " << m_calls << std::endl;
        std::cout << TAG << "Chi2 total: " << chi2_stat + chi2_sys + chi2_reg << std::endl;
        std::cout << TAG << "Chi2 stat : " << chi2_stat << std::endl
                  << TAG << "Chi2 syst : " << chi2_sys  << std::endl
                  << TAG << "Chi2 reg  : " << chi2_reg  << std::endl;
    }

    // The total chi2 value is returned:
    return chi2_stat + chi2_sys + chi2_reg;
}

void XsecFitter::SaveEventHist(int fititer, bool is_final)
{
    for(int s = 0; s < m_samples.size(); s++)
    {
        std::stringstream ss;
        ss << "evhist_sam" << s;
        if(is_final)
            ss << "_finaliter";
        else
            ss << "_iter" << m_calls;

        m_samples[s]->Write(m_dir, ss.str(), fititer);
    }
}

void XsecFitter::SaveEventTree(std::vector<std::vector<double>>& res_params)
{
    outtree = new TTree("selectedEvents", "selectedEvents");
    InitOutputTree();

    for(size_t s = 0; s < m_samples.size(); s++)
    {
        for(int i = 0; i < m_samples[s]->GetN(); i++)
        {
            AnaEvent* ev = m_samples[s]->GetEvent(i);
            ev->SetEvWght(ev->GetEvWghtMC());
            for(size_t j = 0; j < m_fitpara.size(); j++)
            {
                const std::string det = m_samples[s]->GetDetector();
                m_fitpara[j]->ReWeight(ev, det, s, i, res_params[j]);
            }

            sample   = ev->GetSampleType();
            sigtype  = ev->GetSignalType();
            topology = ev->GetTopology();
            reaction = ev->GetReaction();
            target   = ev->GetTarget();
            nutype   = ev->GetFlavor();
            D1true   = ev->GetTrueD1();
            D2true   = ev->GetTrueD2();
            D1Reco   = ev->GetRecoD1();
            D2Reco   = ev->GetRecoD2();
            weightMC = ev->GetEvWghtMC() * m_potratio;
            weight   = ev->GetEvWght() * m_potratio;
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

void XsecFitter::ParameterScans(const std::vector<int>& param_list, unsigned int nsteps)
{
    std::cout << TAG << "Performing parameter scans..." << std::endl;

    //Internally Scan performs steps-1, so add one to actually get the number of steps
    //we ask for.
    unsigned int adj_steps = nsteps+1;
    double* x = new double[adj_steps] {};
    double* y = new double[adj_steps] {};

    for(const auto& p : param_list)
    {
        std::cout << TAG << "Scanning parameter " << p
                  << " (" << m_fitter->VariableName(p) << ")." << std::endl;

        bool success = m_fitter->Scan(p, adj_steps, x, y);

        TGraph scan_graph(nsteps, x, y);
        m_dir->cd();

        std::stringstream ss;
        ss << "par_scan_" << std::to_string(p);
        scan_graph.Write(ss.str().c_str());
    }

    delete[] x;
    delete[] y;
}

TMatrixD XsecFitter::GetPriorCovarianceMatrix(){

  std::vector<TMatrixD*> matrix_category_list;
  int nb_dof = 0;
  for(int i_parameter = 0 ; i_parameter < m_fitpara.size() ; i_parameter++){
    nb_dof += m_fitpara[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  TMatrixD covMatrix(nb_dof,nb_dof);

  int index_shift = 0;
  for(int i_parameter = 0 ; i_parameter < m_fitpara.size() ; i_parameter++){
    for(int i_entry = 0 ; i_entry < m_fitpara[i_parameter]->GetOriginalCovMat()->GetNrows() ; i_entry++){
      for(int j_entry = 0 ; j_entry < m_fitpara[i_parameter]->GetOriginalCovMat()->GetNrows() ; j_entry++){
        covMatrix[i_entry+index_shift][j_entry+index_shift] = (*m_fitpara[i_parameter]->GetOriginalCovMat())[i_entry][j_entry] ;
      }
    }
    index_shift += m_fitpara[i_parameter]->GetOriginalCovMat()->GetNrows();
  }

  return covMatrix;

}

TMatrixD XsecFitter::GetPosteriorCovarianceMatrix(){

  TMatrixD covMatrix(m_fitter->NDim(), m_fitter->NDim() );
  m_fitter->GetCovMatrix( covMatrix.GetMatrixArray() );
  return covMatrix;

}

void XsecFitter::WriteCovarianceMatrices(){

  std::cout << TAG << "Writing Covariance Matrices..." << std::endl;
  m_dir->WriteTObject(GetPriorCovarianceMatrix().Clone(), "PriorCovarianceMatrix_TMatrixD");
  m_dir->WriteTObject(GetPosteriorCovarianceMatrix().Clone(), "PosteriorCovarianceMatrix_TMatrixD");

}
