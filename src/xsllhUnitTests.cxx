#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "ColorOutput.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsUnitTest]: " + color::RESET_STR;
    std::cout << "---------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Unit Testing Interface.\n"
              << TAG << "Initializing the testing machinery..." << std::endl;

    return 0;
}
