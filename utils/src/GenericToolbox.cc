//
// Created by Adrien BLANCHET on 02/04/2020.
//

#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <TColor.h>
#include <TError.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TMatrixD.h>
#include <TMatrixDSymEigen.h>
#include <TH1D.h>

#include "GenericToolbox.h"

namespace GenericToolbox
{

static double last_displayed_value = -1;
static int verbosity_level         = 0;
static Color_t colorCells[10]      = {kOrange + 1, kGreen - 3, kTeal + 3,   kAzure + 7, kCyan - 2,
                                 kBlue - 7,   kBlue + 2,  kOrange + 9, kRed + 2,   kPink + 9};

void display_loading(int current_index_, int end_index_, std::string title_, bool force_display_)
{

    int percent = int(std::round(double(current_index_) / end_index_ * 100.));
    if(force_display_ or current_index_ >= end_index_ - 1)
    {
        if(last_displayed_value != -1)
            std::clog << "\r" << title_ << " : " << 100 << "%" << std::endl;
        last_displayed_value = -1;
        return;
    }
    if(last_displayed_value == -1 or last_displayed_value < percent)
    {
        last_displayed_value = percent;
        std::clog << "\r" << title_ << " : " << percent << "%" << std::flush << "\r";
    }
}
void write_string_in_file(std::string file_path_, std::string string_to_write_)
{
    std::ofstream out(file_path_.c_str());
    out << string_to_write_;
    out.close();
}

static Int_t old_verbosity = -1;
void toggle_quiet_root()
{

    if(old_verbosity == -1)
    {
        old_verbosity     = gErrorIgnoreLevel;
        gErrorIgnoreLevel = kFatal;
    }
    else
    {
        gErrorIgnoreLevel = old_verbosity;
        old_verbosity     = -1;
    }
}

bool does_element_is_in_vector(int element_, std::vector<int>& vector_)
{
    for(auto const& element : vector_)
    {
        if(element == element_)
            return true;
    }
    return false;
}
bool does_element_is_in_vector(double element_, std::vector<double>& vector_)
{
    for(auto const& element : vector_)
    {
        if(element == element_)
            return true;
    }
    return false;
}
bool does_element_is_in_vector(std::string element_, std::vector<std::string>& vector_)
{
    for(auto const& element : vector_)
    {
        if(element == element_)
            return true;
    }
    return false;
}

bool do_string_contains_substring(std::string string_, std::string substring_)
{
    if(substring_.size() > string_.size())
        return false;
    return string_.find(substring_) != std::string::npos;
}
bool do_string_starts_with_substring(std::string string_, std::string substring_)
{
    if(substring_.size() > string_.size())
        return false;
    return (not string_.compare(0, substring_.size(), substring_));
}
bool do_string_ends_with_substring(std::string string_, std::string substring_)
{
    if(substring_.size() > string_.size())
        return false;
    return (not string_.compare(string_.size() - substring_.size(), substring_.size(), substring_));
}

bool do_path_is_file(std::string file_path_)
{
    struct stat info{};
    stat(file_path_.c_str(), &info);
    return S_ISREG(info.st_mode);
}
bool do_tfile_is_valid(std::string input_file_path_)
{
    bool file_is_valid = false;
    if(do_path_is_file(input_file_path_))
    {
        auto old_verbosity = gErrorIgnoreLevel;
        gErrorIgnoreLevel  = kFatal;
        auto* input_tfile  = TFile::Open(input_file_path_.c_str(), "READ");
        if(do_tfile_is_valid(input_tfile))
        {
            file_is_valid = true;
            input_tfile->Close();
        }
        delete input_tfile;
        gErrorIgnoreLevel = old_verbosity;
    }
    return file_is_valid;
}
bool do_tfile_is_valid(TFile* input_tfile_, bool check_if_writable_)
{

    if(input_tfile_ == nullptr)
    {
        if(verbosity_level >= 1)
            std::cerr << ERROR << "input_tfile_ is a nullptr" << std::endl;
        return false;
    }

    if(not input_tfile_->IsOpen())
    {
        if(verbosity_level >= 1)
            std::cerr << ERROR << "input_tfile_ = " << input_tfile_->GetName() << " is not opened."
                      << std::endl;
        if(verbosity_level >= 1)
            std::cerr << ERROR << "input_tfile_->IsOpen() = " << input_tfile_->IsOpen()
                      << std::endl;
        return false;
    }

    if(check_if_writable_ and not input_tfile_->IsWritable())
    {
        if(verbosity_level >= 1)
            std::cerr << ERROR << "input_tfile_ = " << input_tfile_->GetName()
                      << " is not writable." << std::endl;
        if(verbosity_level >= 1)
            std::cerr << ERROR << "input_tfile_->IsWritable() = " << input_tfile_->IsWritable()
                      << std::endl;
        return false;
    }

    return true;
}

std::string join_vector_string(std::vector<std::string> string_list_, std::string delimiter_,
                               int begin_index_, int end_index_)
{

    std::string joined_string;
    if(end_index_ == 0)
        end_index_ = int(string_list_.size());

    // circular permutation -> python style : tab[-1] = tab[tab.size - 1]
    if(end_index_ < 0 and int(string_list_.size()) > std::fabs(end_index_))
        end_index_ = int(string_list_.size()) + end_index_;

    for(int i_list = begin_index_; i_list < end_index_; i_list++)
    {
        if(not joined_string.empty())
            joined_string += delimiter_;
        joined_string += string_list_[i_list];
    }

    return joined_string;
}
std::string get_current_working_folder_path()
{

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    return cwd;
}
std::string to_lower_case(std::string& input_str_)
{
    std::string output_str(input_str_);
    std::transform(output_str.begin(), output_str.end(), output_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return output_str;
}

std::vector<std::string> read_file(std::string file_path_)
{

    std::vector<std::string> output_vector;
    std::ifstream infile(file_path_);

    std::string line;
    while(std::getline(infile, line))
    {
        output_vector.emplace_back(line);
    }

    infile.close();
    return output_vector;
}

std::vector<std::string> split_string(std::string& input_string_, std::string delimiter_)
{

    std::vector<std::string> output_splited_string;

    const char* src  = input_string_.c_str();
    const char* next = src;

    std::string out_string_piece;

    while((next = strstr(src, delimiter_.c_str())) != NULL)
    {
        out_string_piece = "";
        while(src != next)
        {
            out_string_piece += *src++;
        }
        output_splited_string.emplace_back(out_string_piece);
        /* Skip the delimiter_ */
        src += delimiter_.size();
    }

    /* Handle the last token */
    out_string_piece = "";
    while(*src != '\0')
        out_string_piece += *src++;

    output_splited_string.emplace_back(out_string_piece);

    return output_splited_string;
}

TMatrixDSym* convert_to_symmetric_matrix(TMatrixD* matrix_)
{

    auto* symmetric_matrix            = (TMatrixD*)matrix_->Clone();
    auto* transposed_symmetric_matrix = new TMatrixD(*matrix_);

    transposed_symmetric_matrix->Transpose(*matrix_);
    *symmetric_matrix += *transposed_symmetric_matrix;
    for(int i_col = 0; i_col < matrix_->GetNcols(); i_col++)
    {
        for(int i_row = 0; i_row < matrix_->GetNrows(); i_row++)
        {
            (*symmetric_matrix)[i_row][i_col] /= 2.;
        }
    }

    auto* result = (TMatrixDSym*)symmetric_matrix->Clone(); // Convert to TMatrixDSym

    delete transposed_symmetric_matrix;
    delete symmetric_matrix;

    return result;
}
std::map<std::string, TMatrixD*> SVD_matrix_inversion(TMatrixD* matrix_, std::string output_content_)
{
    std::map<std::string, TMatrixD*> results_handler;

    auto content_names = GenericToolbox::split_string(output_content_, ":");

    if(does_element_is_in_vector("inverse_covariance_matrix", content_names)){
        results_handler["inverse_covariance_matrix"]
            = new TMatrixD(matrix_->GetNrows(), matrix_->GetNcols());
    }
    if(does_element_is_in_vector("regularized_covariance_matrix", content_names)){
        results_handler["regularized_covariance_matrix"]
            = new TMatrixD(matrix_->GetNrows(), matrix_->GetNcols());
    }
    if(does_element_is_in_vector("projector", content_names)){
        results_handler["projector"]
            = new TMatrixD(matrix_->GetNrows(), matrix_->GetNcols());
    }
    if(does_element_is_in_vector("regularized_eigen_values", content_names)){
        results_handler["regularized_eigen_values"]
            = new TMatrixD(matrix_->GetNrows(), 1);
    }

    // make sure all are 0
    for(const auto& matrix_handler : results_handler){
        for(int i_dof = 0; i_dof < matrix_handler.second->GetNrows(); i_dof++){
            for(int j_dof = 0; j_dof < matrix_handler.second->GetNcols(); j_dof++){
                (*matrix_handler.second)[i_dof][j_dof] = 0.;
            }
        }
    }


    // Covariance matrices are symetric :
    auto* symmetric_matrix        = convert_to_symmetric_matrix(matrix_);
    auto* Eigen_matrix_decomposer = new TMatrixDSymEigen(*symmetric_matrix);
    auto* Eigen_values            = &(Eigen_matrix_decomposer->GetEigenValues());
    auto* Eigen_vectors           = &(Eigen_matrix_decomposer->GetEigenVectors());

    double max_eigen_value = (*Eigen_values)[0];
    for(int i_eigen_value = 0; i_eigen_value < matrix_->GetNcols(); i_eigen_value++){
        if(max_eigen_value < (*Eigen_values)[i_eigen_value]){
            max_eigen_value = (*Eigen_values)[i_eigen_value];
        }
    }

    for(int i_eigen_value = 0; i_eigen_value < matrix_->GetNcols(); i_eigen_value++)
    {
        if((*Eigen_values)[i_eigen_value] > max_eigen_value*1E-5)
        {
            if(results_handler.find("regularized_eigen_values") != results_handler.end()){
                (*results_handler["regularized_eigen_values"])[i_eigen_value][0]
                    = (*Eigen_values)[i_eigen_value];
            }
            for(int i_dof = 0; i_dof < matrix_->GetNrows(); i_dof++){
                for(int j_dof = 0; j_dof < matrix_->GetNrows(); j_dof++){
                    if(results_handler.find("inverse_covariance_matrix") != results_handler.end()){
                        (*results_handler["inverse_covariance_matrix"])[i_dof][j_dof]
                            += (1. / (*Eigen_values)[i_eigen_value])
                               * (*Eigen_vectors)[i_dof][i_eigen_value]
                               * (*Eigen_vectors)[j_dof][i_eigen_value];
                    }
                    if(results_handler.find("projector") != results_handler.end()){
                        (*results_handler["projector"])[i_dof][j_dof]
                            += (*Eigen_vectors)[i_dof][i_eigen_value]
                               * (*Eigen_vectors)[j_dof][i_eigen_value];
                    }
                    if(results_handler.find("regularized_covariance_matrix") != results_handler.end()){
                        (*results_handler["regularized_covariance_matrix"])[i_dof][j_dof]
                            += (*Eigen_values)[i_eigen_value]
                               * (*Eigen_vectors)[i_dof][i_eigen_value]
                               * (*Eigen_vectors)[j_dof][i_eigen_value];
                    }

                }
            }
        }
        else{
//            std::cout << ALERT << "Skipping i_eigen_value = " << (*Eigen_values)[i_eigen_value]
//                      << std::endl;
        }
    }

    // No memory leak ? : CHECKED
    delete Eigen_matrix_decomposer;
    delete symmetric_matrix;

    return results_handler;
}

TH1D* get_TH1D_from_TVectorD(std::string graph_title_, TVectorD* Y_values_, std::string Y_title_,
                             std::string X_title_, TVectorD* Y_errors_)
{

    auto* th1_histogram = new TH1D(graph_title_.c_str(), graph_title_.c_str(),
                                   Y_values_->GetNrows(), -0.5, Y_values_->GetNrows() - 0.5);

    for(int i_row = 0; i_row < Y_values_->GetNrows(); i_row++)
    {
        th1_histogram->SetBinContent(i_row + 1, (*Y_values_)[i_row]);
        if(Y_errors_ != nullptr)
            th1_histogram->SetBinError(i_row + 1, (*Y_errors_)[i_row]);
    }

    th1_histogram->SetLineWidth(2);
    th1_histogram->SetLineColor(kBlue);
    th1_histogram->GetXaxis()->SetTitle(X_title_.c_str());
    th1_histogram->GetYaxis()->SetTitle(Y_title_.c_str());

    return th1_histogram;
}
TH2D* get_TH2D_from_TMatrixD(TMatrixD* XY_values_, std::string graph_title_, std::string Z_title_,
                             std::string Y_title_, std::string X_title_)
{

    if(graph_title_.empty())
        graph_title_ = XY_values_->GetTitle();

    auto* th2_histogram = new TH2D(graph_title_.c_str(), graph_title_.c_str(),
                                   XY_values_->GetNrows(), -0.5, XY_values_->GetNrows() - 0.5,
                                   XY_values_->GetNcols(), -0.5, XY_values_->GetNcols() - 0.5);

    for(int i_col = 0; i_col < XY_values_->GetNcols(); i_col++)
    {
        for(int j_row = 0; j_row < XY_values_->GetNrows(); j_row++)
        {
            th2_histogram->SetBinContent(i_col + 1, j_row + 1, (*XY_values_)[i_col][j_row]);
        }
    }

    th2_histogram->GetXaxis()->SetTitle(X_title_.c_str());
    th2_histogram->GetYaxis()->SetTitle(Y_title_.c_str());
    th2_histogram->GetZaxis()->SetTitle(Z_title_.c_str());

    return th2_histogram;
}

TVectorD *get_TVectorD_from_vector(std::vector<double>& vect_){

    auto *output = new TVectorD(vect_.size());
    for(int i = 0 ; i < int(vect_.size()) ; i++){
        (*output)[i] = vect_[i];
    }
    return output;

}

} // namespace GenericToolbox
