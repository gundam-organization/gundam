void make_flux_plots(const std::string& input_filename)
{
    TFile* file = TFile::Open(input_filename.c_str(), "READ");

    gStyle->SetPadTopMargin(0.10);
    gStyle->SetPadRightMargin(0.10);
    gStyle->SetPadLeftMargin(0.14);
    gStyle->SetPadBottomMargin(0.14);
    gStyle->SetLabelSize(0.050, "xyzt");
    gStyle->SetLabelOffset(0.010, "xyzt");
    gStyle->SetTitleSize(0.060, "xyzt");
    gStyle->SetTitleOffset(1.10, "xyzt");
    gStyle->SetStripDecimals(false);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetTitleX(0.5);
    gStyle->SetTitleAlign(23);
    gStyle->SetNdivisions(505);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    double flux_scaling = 1.0 / 0.58;
    TH1D* flux_numu = (TH1D*)file->Get("flux_fine_numu");
    flux_numu->Scale(flux_scaling);
    TH1D* flux_numubar = (TH1D*)file->Get("flux_fine_numubar");
    flux_numubar->Scale(flux_scaling);
    TH1D* flux_nue = (TH1D*)file->Get("flux_fine_nue");
    flux_nue->Scale(flux_scaling);
    TH1D* flux_nuebar = (TH1D*)file->Get("flux_fine_nuebar");
    flux_nuebar->Scale(flux_scaling);

    /*
    TH1D* flux_numu = (TH1D*)file->Get("nd2_tune_numu");
    flux_numu->UseCurrentStyle();
    TH1D* flux_numubar = (TH1D*)file->Get("nd2_tune_numub");
    flux_numubar->UseCurrentStyle();
    TH1D* flux_nue = (TH1D*)file->Get("nd2_tune_nue");
    flux_nue->UseCurrentStyle();
    TH1D* flux_nuebar = (TH1D*)file->Get("nd2_tune_nueb");
    flux_nuebar->UseCurrentStyle();
    */

    TCanvas* c = new TCanvas("c","c",1200,900);
    c->SetLogy(1);
    c->SetGridy(1);

    flux_numu->GetXaxis()->SetRangeUser(0.0, 10.0);
    flux_numu->GetYaxis()->SetRangeUser(1E8, 3E12);
    flux_numu->GetXaxis()->SetTitle("E_{#nu} (GeV)");
    flux_numu->GetYaxis()->SetTitle("Flux (/cm^{2}/50 MeV/10^{21} POT)");
    flux_numu->SetLineWidth(2);
    flux_numu->SetLineColor(kRed+1);
    flux_numu->Draw("hist e");
    flux_numubar->SetLineWidth(2);
    flux_numubar->SetLineColor(kOrange+1);
    flux_numubar->Draw("hist e same");
    flux_nue->SetLineWidth(2);
    flux_nue->SetLineColor(kBlue+2);
    flux_nue->Draw("hist e same");
    flux_nuebar->SetLineWidth(2);
    flux_nuebar->SetLineColor(kAzure-4);
    flux_nuebar->Draw("hist e same");

    TLegend* l = new TLegend(0.7, 0.6, 0.9, 0.8);
    l->AddEntry(flux_numu, "#nu_{#mu}", "l");
    l->AddEntry(flux_numubar, "#bar{#nu}_{#mu}", "l");
    l->AddEntry(flux_nue, "#nu_{e}", "l");
    l->AddEntry(flux_nuebar, "#bar{#nu}_{e}", "l");
    l->SetTextSize(0.06);
    l->SetBorderSize(0);
    l->SetFillStyle(0);
    l->Draw("same");

    c->Print("nd280_nominal_flux_prediction.pdf");
    //c->Print("ingrid_nominal_flux_prediction.pdf");

    return;
}
