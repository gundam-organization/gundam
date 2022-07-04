#include "CalcChisq.hh"

CalcChisq::CalcChisq()
    : did_invert(false)
    , tol(1E-100)
    , npar(0)
    , cov_mat(nullptr)
    , inv_mat(nullptr)
{
}

CalcChisq::CalcChisq(const TMatrixD& cov)
    : CalcChisq()
{
    SetCovariance(cov);
}

CalcChisq::CalcChisq(const TMatrixDSym& cov)
    : CalcChisq()
{
    SetCovariance(cov);
}

CalcChisq::~CalcChisq()
{
    if(cov_mat != nullptr)
        delete cov_mat;

    if(inv_mat != nullptr)
        delete inv_mat;
}

void CalcChisq::SetCovariance(const TMatrixD& cov)
{
    if(cov_mat != nullptr)
        delete cov_mat;

    if(inv_mat != nullptr)
        delete inv_mat;

    npar = cov.GetNrows();
    cov_mat = new TMatrixD(npar, npar);

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*cov_mat)[i][j] = cov[i][j];
        }
    }

    inv_mat = new TMatrixD(*cov_mat);
    double det = 0;
    if(!TDecompLU::InvertLU(*inv_mat, tol, &det))
    {
        std::cout << ERR << "Failed to invert matrix." << std::endl;
        did_invert = false;
    }
    else
    {
        std::cout << TAG << "Inversion successful." << std::endl;
        did_invert = true;
    }
}

void CalcChisq::SetCovariance(const TMatrixDSym& cov)
{
    if(cov_mat != nullptr)
        delete cov_mat;

    if(inv_mat != nullptr)
        delete inv_mat;

    npar = cov.GetNrows();
    cov_mat = new TMatrixD(npar, npar);

    for(int i = 0; i < npar; ++i)
    {
        for(int j = 0; j < npar; ++j)
        {
            (*cov_mat)[i][j] = cov[i][j];
        }
    }

    inv_mat = new TMatrixD(*cov_mat);
    double det = 0;
    if(!TDecompLU::InvertLU(*inv_mat, tol, &det))
    {
        std::cout << ERR << "Failed to invert matrix." << std::endl;
        did_invert = false;
    }
    else
    {
        std::cout << TAG << "Inversion successful." << std::endl;
        did_invert = true;
    }
}

double CalcChisq::CalcChisqCov(const TH1D& h1, const TH1D& h2) const
{
    double chisq = 0;

    if(!did_invert)
        return chisq;
    else
    {
        if(h1.GetNbinsX() != h2.GetNbinsX())
        {
            std::cout << ERR << "Histogram bins do not match!" << std::endl;
            return chisq;
        }

        if(h1.GetNbinsX() != npar)
        {
            std::cout << ERR << "Histogram bins do not match covariance parameters!" << std::endl;
            return chisq;
        }

        for(int i = 0; i < npar; ++i)
        {
            for(int j = 0; j < npar; ++j)
            {
                double x = h1.GetBinContent(i+1) - h2.GetBinContent(i+1);
                double y = h1.GetBinContent(j+1) - h2.GetBinContent(j+1);
                chisq += x * y * (*inv_mat)[i][j];
            }
        }
    }

    return chisq;
}

double CalcChisq::CalcChisqStat(const TH1D& h1, const TH1D& h2) const
{
    double chisq = 0;
    if(h1.GetNbinsX() != h2.GetNbinsX())
    {
        std::cout << ERR << "Histogram bins do not match!" << std::endl;
        return chisq;
    }

    for(unsigned int i = 0; i < h1.GetNbinsX(); ++i)
    {
        double observed = h1.GetBinContent(i+1);
        double expected = h2.GetBinContent(i+1);

        if(expected > 0.0)
        {
            chisq += 2.0 * (expected - observed);
            if(observed > 0.0)
                chisq += 2.0 * observed * TMath::Log(observed / expected);
        }
    }

    if(chisq < 0.0)
    {
        std::cout << TAG << "Chisquare is negative. Setting to zero." << std::endl;
        chisq = 0.0;
    }

    if(chisq != chisq)
    {
        std::cout << TAG << "Chisquare is NaN. Setting to zero." << std::endl;
        chisq = 0.0;
    }

    return chisq;
}
