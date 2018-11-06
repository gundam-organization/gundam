#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "TDecompLU.h"
#include "TFile.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"

#include "ColorOutput.hh"
using TMatrixD = TMatrixT<double>;
using TMatrixDSym = TMatrixTSym<double>;

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsXsecCov]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR
                            + "[ERROR]: " + color::RESET_STR;

    std::cout << "--------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Xsec Covariance Builder.\n"
              << TAG << "Initializing the building machinery..." << std::endl;

    double offset = 0.0;
    bool do_ingrid = false;
    bool do_sym_matrix = true;
    bool do_correlation = false;
    std::string fname_input;
    std::string fname_output = "xsec_cov.root";
    std::string matrix_name = "xsec";
    std::string row_mask_str;
    std::vector<int> row_mask;

    char option;
    while((option = getopt(argc, argv, "i:o:m:b:r:CISh")) != -1)
    {
        switch(option)
        {
            case 'i':
                fname_input = optarg;
                break;
            case 'o':
                fname_output = optarg;
                break;
            case 'm':
                matrix_name = optarg;
                break;
            case 'b':
                offset = std::stod(optarg);
                break;
            case 'r':
                row_mask_str = optarg;
                break;
            case 'C':
                do_correlation = true;
                break;
            case 'I':
                do_ingrid = true;
                break;
            case 'S':
                do_sym_matrix = false;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS\n"
                          << "-i : Input xsec file (.txt)\n"
                          << "-o : Output ROOT filename\n"
                          << "-m : Covariance matrix name\n"
                          << "-b : Add value to diagonal\n"
                          << "-C : Calculate correlation matrix\n"
                          << "-I : Build INGRID covariance\n"
                          << "-S : Store as TMatrixT\n"
                          << "-h : Display this help message\n";
            default:
                return 0;
        }
    }

    std::cout << TAG << "Reading covariance text file: " << fname_input << std::endl;
    std::ifstream fin(fname_input, std::ios::in);

    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    std::cout << TAG << "Opening output file: " << fname_output << std::endl;

    TMatrixDSym xsec_cov;
    if(!fin.is_open())
    {
        std::cerr << ERR << "Failed to open " << fname_input << std::endl;
        return 1;
    }

    unsigned int dim = 0;
    std::string line;
    if(std::getline(fin, line))
    {
        std::stringstream ss(line);
        ss >> dim;
    }

    xsec_cov.ResizeTo(dim, dim);
    for(unsigned int i = 0; i < dim; ++i)
    {
        std::getline(fin, line);
        std::stringstream ss(line);
        double val = 0;

        for(unsigned int j = 0; j < dim; ++j)
        {
            ss >> val;
            xsec_cov(i,j) = val;
        }
    }

    if(offset > 0.0)
    {
        std::cout << TAG << "Adding " << offset << " to diagonal of matrix." << std::endl;
        for(unsigned int i = 0; i < dim; ++i)
            xsec_cov(i,i) += offset;
    }

    if(!row_mask_str.empty())
    {
        std::stringstream ss(row_mask_str);
        for(std::string s; std::getline(ss, s, ',');)
            row_mask.emplace_back(std::stoi(s));

        std::cout << TAG << "Masking the following rows: ";
        for(const auto& val : row_mask)
            std::cout << val << " ";
        std::cout << std::endl;

        std::vector<bool> row_mask_bool(dim, false);
        for(const auto& row : row_mask)
            row_mask_bool.at(row) = true;

        const unsigned int dim_reduce = dim - row_mask.size();
        unsigned int i_r{0}, j_r{0};

        TMatrixDSym xsec_cov_reduce(dim_reduce);
        for(unsigned int i = 0; i < dim; ++i)
        {
            if(!row_mask_bool[i])
            {
                for(unsigned int j = 0; j < dim; ++j)
                {
                    if(!row_mask_bool[j])
                    {
                        xsec_cov_reduce(i_r,j_r) = xsec_cov(i,j);
                        j_r++;
                    }
                }
                i_r++;
                j_r = 0;
            }
        }

        dim = dim_reduce;
        xsec_cov.ResizeTo(dim, dim);
        xsec_cov = xsec_cov_reduce;
    }

    TDecompLU inv_test;
    TMatrixD xsec_inv(xsec_cov);
    if(inv_test.InvertLU(xsec_inv, 1E-40))
    {
        std::cout << TAG << "Matrix is invertible." << std::endl;
        std::string name = matrix_name + "_cov";
        xsec_cov.Write(name.c_str());
    }

    if(do_correlation)
    {
        std::cout << TAG << "Calculation correlation matrix." << std::endl;
        TMatrixDSym xsec_cor(dim);
        for(unsigned int i = 0; i < dim; ++i)
        {
            for(unsigned int j = 0; j < dim; ++j)
            {
                const double x = xsec_cov(i,i);
                const double y = xsec_cov(j,j);
                xsec_cor(i,j) = xsec_cov(i,j) / std::sqrt(x*y);
            }
        }
        const std::string name = matrix_name + "_cor";
        xsec_cor.Write(name.c_str());
    }

    if(do_ingrid)
    {
        std::cout << TAG << "Calculating INGRID matrix." << std::endl;
        const unsigned int dim_multi = dim * 2;
        TMatrixDSym xsec_cov_ingrid(dim_multi);
        for(unsigned int i = 0; i < dim; ++i)
        {
            for(unsigned int j = 0; j < dim; ++j)
            {
                xsec_cov_ingrid(i,j) = xsec_cov(i,j);
                xsec_cov_ingrid(i,j+dim) = xsec_cov(i,j);
                xsec_cov_ingrid(i+dim,j) = xsec_cov(i,j);
                xsec_cov_ingrid(i+dim,j+dim) = xsec_cov(i,j);
            }
        }

        if(offset > 0.0)
        {
            std::cout << TAG << "Adding " << offset << " to diagonal of matrix." << std::endl;
            for(unsigned int i = 0; i < dim; ++i)
                xsec_cov_ingrid(i,i) += offset;
        }

        TMatrixD xsec_inv_ingrid(xsec_cov_ingrid);
        if(inv_test.InvertLU(xsec_inv_ingrid, 1E-40))
        {
            std::cout << TAG << "Matrix is invertible." << std::endl;
            std::string name = matrix_name + "_ingrid_cov";
            xsec_cov_ingrid.Write(name.c_str());
        }

        if(do_correlation)
        {
            std::cout << TAG << "Calculation correlation matrix." << std::endl;
            TMatrixDSym xsec_cor(dim_multi);
            for(unsigned int i = 0; i < dim_multi; ++i)
            {
                for(unsigned int j = 0; j < dim_multi; ++j)
                {
                    double x = xsec_cov_ingrid(i,i);
                    double y = xsec_cov_ingrid(j,j);
                    xsec_cor(i,j) = xsec_cov_ingrid(i,j) / std::sqrt(x*y);
                }
            }
            std::string name = matrix_name + "_ingrid_cor";
            xsec_cor.Write(name.c_str());
        }
    }

    file_output -> Close();

    std::cout << TAG << "Finished. Saved covariance matrix." << std::endl;
    return 0;
}
