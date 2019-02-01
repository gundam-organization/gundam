void weight_flux(const std::string& flux_path)
{
    const double run2_pot = 0.079263;
    const double run3b_pot = 0.021727;
    const double run3c_pot = 0.136447;
    const double run4_pot = 0.342548;
    const double total = run2_pot + run3b_pot + run3c_pot + run4_pot;

    std::cout << "POT Info (10^21)" << std::endl;
    std::cout << "Run2 : " << run2_pot << std::endl;
    std::cout << "Run3b: " << run3b_pot << std::endl;
    std::cout << "Run3c: " << run3c_pot << std::endl;
    std::cout << "Run4 : " << run4_pot << std::endl;
    std::cout << "Total: " << total << std::endl;

    std::string run2_file = flux_path + "/run2/nd5_tuned13av2_13anom_run2_numode_fine.root";
    TFile* flux_run2 = TFile::Open(run2_file.c_str(), "READ");
    TH1D* h_flux_run2 = (TH1D*)flux_run2 -> Get("enu_nd5_tuned13a_numu");
    h_flux_run2 -> Scale(run2_pot);

    std::string run3b_file = flux_path + "/run3b/nd5_tuned13av2_13anom_run3b_numode_fine.root";
    TFile* flux_run3b = TFile::Open(run3b_file.c_str(), "READ");
    TH1D* h_flux_run3b = (TH1D*)flux_run3b -> Get("enu_nd5_tuned13a_numu");
    h_flux_run3b -> Scale(run3b_pot);

    std::string run3c_file = flux_path + "/run3c/nd5_tuned13av2_13anom_run3c_numode_fine.root";
    TFile* flux_run3c = TFile::Open(run3c_file.c_str(), "READ");
    TH1D* h_flux_run3c = (TH1D*)flux_run3c -> Get("enu_nd5_tuned13a_numu");
    h_flux_run3c -> Scale(run3c_pot);

    std::string run4_file = flux_path + "/run4/nd5_tuned13av2_13anom_run4_numode_fine.root";
    TFile* flux_run4 = TFile::Open(run4_file.c_str(), "READ");
    TH1D* h_flux_run4 = (TH1D*)flux_run4 -> Get("enu_nd5_tuned13a_numu");
    h_flux_run4 -> Scale(run4_pot);

    TH1D* h_flux_total_numu = (TH1D*)h_flux_run2 -> Clone("h_flux_total_numu");
    h_flux_total_numu -> Add(h_flux_run3b);
    h_flux_total_numu -> Add(h_flux_run3c);
    h_flux_total_numu -> Add(h_flux_run4);

    TH1D* h_flux_nom_numu = nullptr;

    //const unsigned int nbins = 11;
    //double flux_bins[nbins+1] = {0.0, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 2.5, 3.5, 5.0, 7.0, 30.0};
    const unsigned int nbins = 20;
    double flux_bins[21] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0,
                            1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 30.0};
    h_flux_nom_numu = (TH1D*)h_flux_total_numu -> Rebin(nbins, "h_flux_nom_numu", flux_bins);

    TFile* output = TFile::Open("./weighted_flux13av2_run2-4.root", "RECREATE");
    output -> cd();
    h_flux_total_numu -> Write("flux_fine");
    h_flux_nom_numu -> Write("flux_rebin");
}
