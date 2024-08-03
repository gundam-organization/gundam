void plot_fit_output(const std::string& file_name_input)
{
    gStyle -> SetOptStat(0);

    TFile* file = TFile::Open(file_name_input.c_str(), "READ");
    std::vector<std::string> par_name = {"par_fit", "par_flux", "par_xsec", "par_det"};

    for(const auto& name : par_name)
    {
        std::stringstream ss;
        ss << "hist_" << name << "_prior";
        TH1D* h_prior = (TH1D*)file -> Get(ss.str().c_str());

        ss.str("");
        ss << "hist_" << name << "_result";
        TH1D* h_final = (TH1D*)file -> Get(ss.str().c_str());

        TCanvas c("c", "c", 1200, 900);
        h_prior -> SetLineColor(kBlack);
        h_prior -> SetLineWidth(2);
        h_final -> SetLineColor(kRed);
        h_final -> SetLineWidth(2);

        h_final -> Draw("hist");
        h_prior -> Draw("hist same");

        ss.str("");
        ss << name << "_results.png";
        c.Print(ss.str().c_str());
    }

    for(const auto& name : par_name)
    {
        std::stringstream ss;

        ss.str("");
        ss << "hist_" << name << "_error_prior";
        TH1D* h_err_prior = (TH1D*)file -> Get(ss.str().c_str());

        ss.str("");
        ss << "hist_" << name << "_error_final";
        TH1D* h_err_final = (TH1D*)file -> Get(ss.str().c_str());

        TCanvas c("c", "c", 1200, 900);
        h_err_prior -> SetLineColor(kBlack);
        h_err_prior -> SetLineWidth(2);
        h_err_final -> SetLineColor(kRed);
        h_err_final -> SetLineWidth(2);

        h_err_final -> Draw("hist");
        h_err_prior -> Draw("hist same");

        ss.str("");
        ss << name << "_errors.png";
        c.Print(ss.str().c_str());
    }

    for(const auto& name : par_name)
    {
        std::stringstream ss;
        ss << "hist_" << name << "_prior";
        TH1D* h_prior = (TH1D*)file -> Get(ss.str().c_str());

        ss.str("");
        ss << "hist_" << name << "_result";
        TH1D* h_final = (TH1D*)file -> Get(ss.str().c_str());

        ss.str("");
        ss << "hist_" << name << "_error_prior";
        TH1D* h_err_prior = (TH1D*)file -> Get(ss.str().c_str());

        ss.str("");
        ss << "hist_" << name << "_error_final";
        TH1D* h_err_final = (TH1D*)file -> Get(ss.str().c_str());

        for(unsigned int i = 1; i <= h_prior->GetNbinsX(); ++i)
        {
            h_prior -> SetBinError(i, h_err_prior -> GetBinContent(i));
            h_final -> SetBinError(i, h_err_final -> GetBinContent(i));
        }

        TCanvas c("c", "c", 1200, 900);
        h_prior -> SetMarkerColor(kBlack);
        h_prior -> SetMarkerStyle(kFullCircle);
        h_final -> SetMarkerColor(kBlue);
        h_final -> SetMarkerStyle(kFullCircle);

        h_prior -> SetFillColor(kRed);
        h_prior -> SetFillStyle(1001);
        h_final -> SetFillColor(kBlue);
        h_final -> SetFillStyle(3144);

        h_prior -> Draw("P E2");
        h_final -> Draw("P E2 same");

        ss.str("");
        ss << name << "_overlay.png";
        c.Print(ss.str().c_str());
    }
}
