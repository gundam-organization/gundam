#include <RVersion.h>
double calc_chisq(std::string filename, bool is_txt_file = false)
{
    if(ROOT_VERSION_CODE < ROOT_VERSION(6,0,0))
    {
        std::cout << "[ERROR]: ROOT version too old! Need ROOT 6 or higher." << std::endl;
        return -1;
    }

    std::vector<std::string> v_files;
    std::string h1_name("sel_best_fit");
    std::string h2_name("tru_best_fit");
    std::string cov_name("xsec_cov");

    if(is_txt_file)
    {
        std::ifstream fin(filename, std::ios::in);
        if(!fin.is_open())
        {
            std::cout << "[ERROR]: Failed to open " << filename << std::endl;
            return -1;
        }
        else
        {
            std::string line;
            while(std::getline(fin, line))
                v_files.emplace_back(line);
        }
    }
    else
        v_files.emplace_back(filename);

    unsigned int dof = 0;
    TH1D* h_chisq = new TH1D("h_chisq", "h_chisq", 50, 0, 300);
    TH2D* h_candle = new TH2D("h_candle", "h_candle", 70, 0, 70, 10000, -2E-39, 15E-39);
    //TH2D* h_errors = new TH2D("h_errors", "h_errors", 70, 0, 70, 10000, 1E-42, 5E-39);
    TH2D* h_errors = new TH2D("h_errors", "h_errors", 70, 0, 70, 1000, 0, 3);

    for(const auto& file : v_files)
    {
        std::cout << "Reading " << file << std::endl;
        TFile* f = TFile::Open(file.c_str(), "READ");
        TH1D* h1 = nullptr;
        TH1D* h2 = nullptr;
        TMatrixDSym* cov_sym = nullptr;

        f->GetObject(h1_name.c_str(), h1);
        f->GetObject(h2_name.c_str(), h2);
        f->GetObject(cov_name.c_str(), cov_sym);

        if(h1->GetNbinsX() != h2->GetNbinsX())
        {
            std::cout << "[ERROR]: Histograms bin numbers do not match!" << std::endl;
            return -1;
        }

        TMatrixD cov(*cov_sym);
        TMatrixD inv(*cov_sym);

        double det = 0;
        if(!TDecompLU::InvertLU(inv, 1E-100, &det))
        {
            std::cout << "[ERROR]: Failed to invert matrix." << std::endl;
            return -1;
        }

        double chisq = 0;
        const unsigned int nbins = cov.GetNrows();
        for(int i = 0; i < nbins; ++i)
        {
            for(int j = 0; j < nbins; ++j)
            {
                double x = h1->GetBinContent(i+1) - h2->GetBinContent(i+1);
                double y = h1->GetBinContent(j+1) - h2->GetBinContent(j+1);
                chisq += x * y * inv[i][j];
            }

            h_candle->Fill(i, h1->GetBinContent(i+1));
            h_errors->Fill(i, h1->GetBinError(i+1)/h1->GetBinContent(i+1));
        }

        if(is_txt_file)
        {
            dof = nbins - 1;
            h_chisq->Fill(chisq);
            std::cout << "Chisq = " << chisq << std::endl;
        }
        else
        {
            std::cout << "Chisq = " << chisq << std::endl;
            return chisq;
        }
    }

    TF1* f_chisq = new TF1("f_chisq", "ROOT::Math::chisquared_pdf(x,[0],0)",0,150);
    f_chisq->SetParameter(0, dof);

    TCanvas* c = new TCanvas("c","c",1200,900);
    h_chisq->Scale(1.0 / h_chisq->Integral());
    h_chisq->Draw("hist");
    f_chisq->Draw("same");

    c->Print("chisq_dist.png");

    TCanvas* v = new TCanvas("v","v",1200,900);
    h_candle->Draw("CANDLEX3");

    TCanvas* e = new TCanvas("e","e",1200,900);
    h_errors->Draw("CANDLEX3");

    return 0;
}
