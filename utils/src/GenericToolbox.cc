//
// Created by Adrien BLANCHET on 02/04/2020.
//

#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <climits>

#include <TColor.h>
#include <TError.h>
#include <TFile.h>

#include "GenericToolbox.h"

namespace GenericToolbox{

static double last_displayed_value = -1;
static int verbosity_level = 0;
static Color_t colorCells[10] = {kOrange+1, kGreen-3, kTeal+3, kAzure+7, kCyan-2, kBlue-7, kBlue+2, kOrange+9, kRed+2, kPink+9};

void display_loading(int current_index_, int end_index_, std::string title_, bool force_display_) {

    int percent = int(std::round(double(current_index_) / end_index_ * 100.));
    if(force_display_ or current_index_ >= end_index_-1) {
        if(last_displayed_value != -1) std::clog << "\r" << title_ << " : " << 100 << "%" << std::endl;
        last_displayed_value = -1;
        return;
    }
    if(last_displayed_value == -1 or last_displayed_value < percent) {
        last_displayed_value = percent;
        std::clog << "\r" << title_ << " : " << percent << "%" << std::flush << "\r";
    }

}
void write_string_in_file(std::string file_path_, std::string string_to_write_){
    std::ofstream out(file_path_.c_str());
    out << string_to_write_;
    out.close();
}

static Int_t old_verbosity = -1;
void toggle_quiet_root(){

    if(old_verbosity == -1){
        old_verbosity = gErrorIgnoreLevel;
        gErrorIgnoreLevel = kFatal;
    } else {
        gErrorIgnoreLevel = old_verbosity;
        old_verbosity = -1;
    }

}

//  template<typename T>
//  bool do_element_in_vector(T element_, std::vector<T>& vector_){
//    for(auto const &element : vector_){
//      if(element == element_) return true;
//    }
//    return false;
//  }

bool does_element_is_in_vector(int element_, std::vector<int>& vector_){
    for(auto const &element : vector_){
        if(element == element_) return true;
    }
    return false;
}
bool does_element_is_in_vector(double element_, std::vector<double>& vector_){
    for(auto const &element : vector_){
        if(element == element_) return true;
    }
    return false;
}
bool does_element_is_in_vector(std::string element_, std::vector<std::string>& vector_){
    for(auto const &element : vector_){
        if(element == element_) return true;
    }
    return false;
}

bool do_string_contains_substring(std::string string_, std::string substring_){
    if(substring_.size() > string_.size()) return false;
    return string_.find(substring_) != std::string::npos;
}
bool do_string_starts_with_substring(std::string string_, std::string substring_){
    if(substring_.size() > string_.size()) return false;
    return (not string_.compare(0, substring_.size(), substring_));
}
bool do_string_ends_with_substring(std::string string_, std::string substring_){
    if(substring_.size() > string_.size()) return false;
    return (not string_.compare(string_.size() - substring_.size(), substring_.size(), substring_));
}

bool do_path_is_file(std::string file_path_){
    struct stat info{};
    stat( file_path_.c_str(), &info );
    return S_ISREG(info.st_mode);
}
bool do_tfile_is_valid(std::string input_file_path_){
    bool file_is_valid = false;
    if(do_path_is_file(input_file_path_)){
        auto old_verbosity = gErrorIgnoreLevel;
        gErrorIgnoreLevel = kFatal;
        auto* input_tfile = TFile::Open(input_file_path_.c_str(), "READ");
        if(do_tfile_is_valid(input_tfile)){
            file_is_valid = true;
            input_tfile->Close();
        }
        delete input_tfile;
        gErrorIgnoreLevel = old_verbosity;
    }
    return file_is_valid;
}
bool do_tfile_is_valid(TFile *input_tfile_, bool check_if_writable_){

    if(input_tfile_ == nullptr){
        if(verbosity_level >= 1) std::cerr << ERROR << "input_tfile_ is a nullptr" << std::endl;
        return false;
    }

    if(not input_tfile_->IsOpen()){
        if(verbosity_level >= 1) std::cerr << ERROR << "input_tfile_ = " << input_tfile_->GetName() << " is not opened." << std::endl;
        if(verbosity_level >= 1) std::cerr << ERROR << "input_tfile_->IsOpen() = " << input_tfile_->IsOpen() << std::endl;
        return false;
    }

    if(check_if_writable_ and not input_tfile_->IsWritable()){
        if(verbosity_level >= 1) std::cerr << ERROR << "input_tfile_ = " << input_tfile_->GetName() << " is not writable." << std::endl;
        if(verbosity_level >= 1) std::cerr << ERROR << "input_tfile_->IsWritable() = " << input_tfile_->IsWritable() << std::endl;
        return false;
    }

    return true;

}

std::string join_vector_string(std::vector<std::string> string_list_, std::string delimiter_, int begin_index_, int end_index_) {

    std::string joined_string;
    if(end_index_ == 0) end_index_ = int(string_list_.size());

    // circular permutation -> python style : tab[-1] = tab[tab.size - 1]
    if(end_index_ < 0 and int(string_list_.size()) > std::fabs(end_index_)) end_index_ = int(string_list_.size()) + end_index_;

    for(int i_list = begin_index_ ; i_list < end_index_ ; i_list++){
        if(not joined_string.empty()) joined_string += delimiter_;
        joined_string += string_list_[i_list];
    }

    return joined_string;
}
std::string get_current_working_folder_path(){

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    return cwd;

}

std::vector<std::string> read_file(std::string file_path_){

    std::vector<std::string> output_vector;
    std::ifstream infile(file_path_);

    std::string line;
    while (std::getline(infile, line))
    {
        output_vector.emplace_back(line);
    }

    infile.close();
    return output_vector;

}

std::vector<std::string> split_string(std::string& input_string_, std::string delimiter_){

    std::vector<std::string> output_splited_string;

    const char *src = input_string_.c_str();
    const char *next = src;

    std::string out_string_piece;

    while ((next = strstr(src, delimiter_.c_str())) != NULL) {
        out_string_piece = "";
        while (src != next){
            out_string_piece += *src++;
        }
        output_splited_string.emplace_back(out_string_piece);
        /* Skip the delimiter_ */
        src += delimiter_.size();
    }

    /* Handle the last token */
    out_string_piece = "";
    while (*src != '\0')
        out_string_piece += *src++;

    output_splited_string.emplace_back(out_string_piece);

    return output_splited_string;

}

}

