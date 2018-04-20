#include "AnaFitParameters.hh"

AnaFitParameters::AnaFitParameters()
{
    m_name    = "none";
    Npar      = 0;
    m_rng_priors = false;
    hasCovMat = false;
    hasRegCovMat = false;

    covariance  = nullptr;
    covarianceI = nullptr;
    throwParms  = nullptr;
}

AnaFitParameters::~AnaFitParameters()
{
    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;
    if(throwParms != nullptr)
        delete throwParms;
}

void AnaFitParameters::SetCovarianceMatrix(const TMatrixDSym& covmat)
{
    if(covariance != nullptr)
        delete covariance;
    if(covarianceI != nullptr)
        delete covarianceI;

    covariance  = new TMatrixDSym(covmat);
    covarianceI = new TMatrixDSym(covmat);
    covarianceI -> SetTol(1e-200);
    double det = covarianceI->Determinant();
    if(abs(det) < 1e-200)
    {
        std::cerr << "[ERROR]: In AnaFitParameters::SetCovarianceMatrix():\n"
                  << "[ERROR]: Covariance matrix is non invertable. Determinant is "
                  << det << std::endl;
        return;
    }
    covarianceI -> Invert(&det);

    std::cout << "[SetCovarianceMatrix]: Covariance matrix size: "
              << covariance -> GetNrows() << " x " << covariance -> GetNrows() << std::endl;

    std::cout << "[SetCovarianceMatrix]: Inverted Cov mat: " << std::endl;
    covarianceI -> Print();
    std::cout << "[SetCovarianceMatrix]: Cov mat: " << std::endl;
    covariance -> Print();

    hasCovMat = true;
}

// SetRegCovarianceMatrix
// This is only implemented for making fake data from nusances.
// This could be the way that regularisation is actually included in the fit but I
// prefer thinking about the summation formulation rather than covariance matricies

// N.B: Every other cov matrix provided is an actual cov matrix, i.e. it has to be inverted
// before being used in a penalty term. The reg cov matrix provided is actually already inverte
// so the provided matrix is the inverted cov matrix and the inverse is the actual cov matrix.
void AnaFitParameters::SetRegCovarianceMatrix(TMatrixDSym *covmat)
{
    //if(hasCovMat) cout << "WARNING: parameter set has both cov mat and reg cov mat!!" << endl;

    if(covariance != nullptr) covariance->Delete();
    if(covarianceI != nullptr) covarianceI->Delete();

    double det;
    covariance  = new TMatrixDSym(*covmat);
    covarianceI = new TMatrixDSym(*covmat);
    covariance->SetTol(1e-200);
    double det_now = covariance->Determinant();
    if(abs(det_now) < 1e-200){
        cout << "Warning,  reg cov matrix is non invertable. Det is:" << endl;
        cout << det_now << endl;
        return;
    }
    (*covariance).Invert(&det);

    cout<<"Number of parameters in reg covariance matrix for "<<m_name
        <<" "<<covariance->GetNrows()<<endl;

    cout << "Inverted reg Cov mat: " << endl;
    covariance->Print();
    cout << "reg Cov mat: " << endl;
    covarianceI->Print();

    hasRegCovMat = true;
}

double AnaFitParameters::GetChi2(const std::vector<double> &params)
{
    if(!hasCovMat) return 0.0;

    if(CheckDims(params) == false)
    {
        std::cout << "[WARNING]: In AnaFitParameters::GetChi2()\n"
                  << "[WARNING]: Warning, dimension check failed." << endl;
        return 0.0;
    }

    double chi2 = 0;
    for(int i=0; i < covarianceI -> GetNrows(); i++)
    {
        for(int j=0; j < covarianceI -> GetNrows(); j++)
        {
            chi2 += (params[i]-pars_prior[i]) * (params[j]-pars_prior[j]) * (*covarianceI)(i,j);
        }
    }

    return chi2;
}

bool AnaFitParameters::CheckDims(const std::vector<double> &params)
{
    bool vector_size = false;
    bool matrix_size = false;

    if(params.size() == pars_prior.size())
    {
        vector_size = true;
    }

    else
    {
        std::cerr << "[ERROR]: Dimension of parameter vector does not match priors.\n"
                  << "[ERROR]: Prams size is: " << params.size() << std::endl
                  << "[ERROR]: Prior size is: " << pars_prior.size() << std::endl;
        vector_size = false;
    }

    if(covariance -> GetNrows() == pars_prior.size())
    {
        matrix_size = true;
    }

    else
    {
        std::cerr << "[ERROR]: Dimension of covariance maxtix does not match priors.\n"
                  << "[ERROR]: Rows in cov mat: " << covariance -> GetNrows() << std::endl
                  << "[ERROR]: Prior size is: " << pars_prior.size() << std::endl;
        matrix_size = false;
    }

    return vector_size && matrix_size;
}

void AnaFitParameters::InitThrows()
{
    cout << "AnaFitParameters::InitThrows" << endl;
    cout << "AnaFitParameters::InitThrows HasCovMat: " << HasCovMat() << endl;
    cout << "AnaFitParameters::InitThrows HasRegCovMat: " << HasRegCovMat() << endl;
    cout << "AnaFitParameters::InitThrows hasCovMat: " << hasCovMat << endl;
    cout << "AnaFitParameters::InitThrows hasRegCovMat: " << hasRegCovMat << endl;
    if(!hasCovMat && !hasRegCovMat) return;
    if(throwParms!=nullptr) delete throwParms;
    TVectorD priorVec(covariance->GetNrows());
    for(int i=0; i<covariance->GetNrows(); i++) priorVec(i) = pars_prior[i];

    throwParms = new ThrowParms(priorVec,(*covariance));
    cout << "AnaFitParameters::InitThrows priorVec:" << endl;
    priorVec.Print();
}

// Do throw
// mode 0 - everything is normal, just do the throw from normal cov mat
// mode 1 - also throw from the regularisation cov mat - for fake data from nuisances
void AnaFitParameters::DoThrow(std::vector<double> &pars, int mode)
{
    pars.clear();
    if(!hasCovMat && mode==0)
    {
        pars = pars_prior;
        return;
    }
    if(!hasCovMat && !hasRegCovMat && mode==1)
    {
        pars = pars_prior;
        return;
    }
    if(mode==3)
    {
        pars = pars_prior;
        return;
    }


    if(!throwParms) InitThrows();

    throwParms->ThrowSet(pars_throw);

    if(((TString)m_name).Contains("flux_shape")){
        flux_mod->Reset();
        for(int j=0; j<Npar;j++){
            //cout<<"parameter "<<j<<" before "<<pars_throw[j]<<endl;
            flux_mod->SetBinContent(j+1,pars_throw[j]*flux->GetBinContent(j+1));
        }
        flux_mod->Scale(flux->Integral()/flux_mod->Integral());
        for(int j=0; j<Npar;j++){
            //cout<<flux_mod->GetBinContent(j+1)<<" - "<<flux->GetBinContent(j+1)<< " / "<<flux->GetBinContent(j+1)<<endl;
            pars_throw[j]=flux_mod->GetBinContent(j+1)/flux->GetBinContent(j+1);
            //cout<<"parameter "<<j<<" after "<<pars_throw[j]<<endl;
        }
    }
    pars = pars_throw;
}

//should go to the flux class? But then how I can use the fluxhisto here?
void AnaFitParameters::SetFluxHisto(TH1F* h_flux)
{
    flux=h_flux;
    if(h_flux->GetXaxis()->GetNbins() != Npar){
        cout<<"Wrong number of flux shape parameters aborting"<<endl;
        abort();
    }
    flux_mod=(TH1F*)(flux->Clone("fluxMod"));
    flux_mod->Reset();
}

void AnaFitParameters::AddDetector(const std::string& det, const std::vector<double>& bins, int offset)
{
    std::cout << "[AnaFitParameters]: Adding detector " << det << std::endl;
    m_det_bins.emplace(std::make_pair(det, bins));
    m_det_offset.emplace(std::make_pair(det, offset));
}
