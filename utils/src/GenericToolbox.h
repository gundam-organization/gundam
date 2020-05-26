//
// Created by Adrien BLANCHET on 02/04/2020.
//

#ifndef XSLLHFITTER_GENERICTOOLBOX_H
#define XSLLHFITTER_GENERICTOOLBOX_H

#include <string>
#include <vector>
#include <sys/stat.h>

namespace GenericToolbox {

static std::string NORMAL = "\033[00m";
static std::string ERROR    = "\033[1;31m[GenericToolbox] \033[00m";
static std::string INFO  = "\033[1;32m[GenericToolbox] \033[00m";
static std::string WARNING   = "\033[1;33m[GenericToolbox] \033[00m";
static std::string ALERT = "\033[1;35m[GenericToolbox] \033[00m";


void display_loading(int current_index_, int end_index_, std::string title_ = "", bool force_display_ = false);
void write_string_in_file(std::string file_path_, std::string string_to_write_);
void toggle_quiet_root();

//  template<typename T>
//  bool do_element_in_vector(T element_, std::vector<T>& vector_);
bool does_element_is_in_vector(int element_, std::vector<int>& vector_);
bool does_element_is_in_vector(double element_, std::vector<double>& vector_);
bool does_element_is_in_vector(std::string element_, std::vector<std::string>& vector_);

bool do_string_contains_substring(std::string string_, std::string substring_);
bool do_string_starts_with_substring(std::string string_, std::string substring_);
bool do_string_ends_with_substring(std::string string_, std::string substring_);

bool do_path_is_file(std::string file_path_);
bool do_tfile_is_valid(std::string input_file_path_);
bool do_tfile_is_valid(TFile *input_tfile_, bool check_if_writable_ = false);

std::string join_vector_string(std::vector<std::string> string_list_, std::string delimiter_, int begin_index_ = 0, int end_index_ = 0);
std::string get_current_working_folder_path();

std::vector<std::string> read_file(std::string file_path_);
std::vector<std::string> split_string(std::string& input_string_, std::string delimiter_);

};


#endif //XSLLHFITTER_GENERICTOOLBOX_H
