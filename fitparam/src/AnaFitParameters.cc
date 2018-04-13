#include "AnaFitParameters.hh"

// ctor
AnaFitParameters::AnaFitParameters()
{
    m_name    = "none";
    Npar      = 0;
    hasCovMat = false;
    hasRegCovMat = false;
    checkDims = false;

    covariance  = nullptr;
    covarianceI = nullptr;
    throwParms  = nullptr;
}

// dtor
AnaFitParameters::~AnaFitParameters()
{
    if(covariance != nullptr) covariance->Delete();
    if(covarianceI != nullptr) covarianceI->Delete();
    if(throwParms != nullptr) delete throwParms;
}

// SetCovarianceMatrix
void AnaFitParameters::SetCovarianceMatrix(const TMatrixDSym& covmat)
{
    if(covariance != nullptr) covariance->Delete();
    if(covarianceI != nullptr) covarianceI->Delete();

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

// GetChi2
double AnaFitParameters::GetChi2(std::vector<double> &params)
{
    //if no covariance matrix ...
    if(!hasCovMat) return 0.0;

    if(!checkDims) //check dimensions of various things are ok
    {
        CheckDims(params);
        cout << "AnaFitParameters.cc: Warning, dimension check failed" << endl;
        if(!checkDims) return 0.0;
    }

    //for(size_t i=0;i<params.size();i++)  cout<<i<<" "<<params[i]<<" "<<pars_prior[i]<<endl;

    double chi2 = 0;
    for(int i=0; i<covarianceI->GetNrows(); i++)
    {
        for(int j=0; j<covarianceI->GetNrows(); j++)
        {
            chi2+= (params[i]-pars_prior[i])*(params[j]-pars_prior[j])*(*covarianceI)(i,j);
        }
    }

    return chi2;
}

// CheckDims
void AnaFitParameters::CheckDims(std::vector<double> &params)
{
    checkDims = (params.size() == pars_prior.size());
    if(!checkDims){
        cerr<<"ERROR: dimensions of vectors don't match"<<endl;
        cout << "Prams size is: " << params.size() << endl;
        cout << "Prior size is: " << pars_prior.size() << endl;
    }
    if(hasCovMat) checkDims = checkDims && (covariance->GetNrows() == (int)pars_prior.size());
    if(!checkDims){
        cerr<<"ERROR: dimensions of vector and cov mat don't match"<<endl;
        cout << "Rows in cov mat: " << covariance->GetNrows() << endl;
        cout << "Prior size: " << pars_prior.size() << endl;
    }
}

// InitThrows
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

