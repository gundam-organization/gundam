void plot_xsec(const std::string& file_name_input)
{
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    TFile* file = TFile::Open(file_name_input.c_str(), "READ");

    std::vector<std::string> v_hist_names
        = {"CC0pi_cos_bin0", "CC0pi_cos_bin1", "CC0pi_cos_bin2", "CC0pi_cos_bin3", "CC0pi_cos_bin4",
           "CC0pi_cos_bin5", "CC0pi_cos_bin6", "CC0pi_cos_bin7", "CC0pi_cos_bin8"};

    for(const auto& name : v_hist_names)
    {
        TH1D* hist = (TH1D*)file->Get(name.c_str());
        TCanvas c("c", "c", 1200, 900);
        c.SetGrid(1,1);

        hist -> SetLineColor(kBlack);
        hist -> SetLineWidth(2);
        hist -> SetMarkerColor(kBlack);
        hist -> SetMarkerStyle(kFullCircle);

        //double xmin = hist -> GetXaxis() -> GetXmin(); std::cout << xmin << std::endl;
        //double xmax = hist -> GetXaxis() -> GetXmax(); std::cout << xmax << std::endl;
        //hist -> GetXaxis() -> SetLimits(xmin / 1000.0, xmax / 1000.0);

        auto root_array = hist -> GetXaxis() -> GetXbins();
        double* xbins = const_cast<double*>(root_array -> GetArray());
        for(int i = 0; i < hist->GetNbinsX()+1; ++i)
            xbins[i] = xbins[i] / 1000.0;
        hist -> GetXaxis() -> Set(hist->GetNbinsX(), xbins);

        hist -> GetXaxis() -> SetTitle("p^{true}_{#mu} (GeV/c)");
        hist -> GetYaxis() -> SetTitle("#frac{d#sigma}{dp_{#mu} dcos#theta_{#mu}} #frac{cm^{2}}{nucleon GeV/c}");
        hist -> Draw("hist e");

        std::string save = name + ".png";
        c.Print(save.c_str());
    }
}
