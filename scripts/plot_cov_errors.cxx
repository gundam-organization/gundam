void plot_cov_errors(const TMatrixTSym<double>& m)
{
    const unsigned int nbins = m.GetNrows();
    TH1D h_err("h_err", "h_err", nbins, 0, nbins);

    for(int i = 0; i < nbins; ++i)
    {
        const double err = TMath::Sqrt(m(i,i));
        h_err.SetBinContent(i+1, err);
    }

    gStyle->SetOptStat(0);
    TCanvas c("c", "c", 1200, 900);
    h_err.SetLineColor(kBlack);
    h_err.SetLineWidth(2);
    h_err.Draw("hist");
    c.Print("errors.png");
}
