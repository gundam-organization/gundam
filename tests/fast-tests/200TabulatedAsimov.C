// Source file to build a library to test the Tabulated dials.
#include <iostream>
#include <cmath>
#include <exception>

#define LIB_NAME "200TabulatedAsimov -- "
#undef LOUD_AND_PROUD

extern "C"
int initializeTable(const char* name, int argc, const char* argv[],
                    int bins) {
#ifdef LOUD_AND_PROUD
    std::cout << LIB_NAME << "INITIALIZE TABLE" << std::endl;
    std::cout << LIB_NAME << "name: " << name << std::endl;
    std::cout << LIB_NAME << "argc: " << argc << std::endl;
    for(int i = 0; i<argc; ++i) {
        std::cout << LIB_NAME << "argv[" << i << "]: " << argv[i]<< std::endl;
    }
    std::cout << LIB_NAME << "bins: " << bins << std::endl;
#endif
    return 100;
}

extern "C"
int updateTable(const char* name,
                double table[], int bins,
                const double par[], int npar) {
#ifdef LOUD_AND_PROUD
    std::cout << LIB_NAME << "UPDATE TABLE" << std::endl;
    std::cout << LIB_NAME << "name: " << name << std::endl;
    std::cout << LIB_NAME << "bins: " << bins << std::endl;
    std::cout << LIB_NAME << "npar: " << npar << std::endl;
    for (int i = 0; i<npar; ++i) {
        std::cout << LIB_NAME << "par[" << i << "]: " << par[i] << std::endl;
    }
#endif
    for (int i = 0; i<npar; ++i) {
        if (not std::isnan(par[i])) continue;
        std::cerr << LIB_NAME << ": " << name
                  << " NAN PARAMETER VALUE: " << i
                  << std::endl;
        throw std::runtime_error("NAN parameter value");
    }
    double amplitude = par[0];
    double period = par[1];
    for (int i = 0; i<bins; ++i) {
        double v = i/10.0;
        table[i] = 1.0 + amplitude*std::cos(2.0*3.1415*v/period);
    }
    return 0;
}

extern "C"
double binTable(const char* name,
                int varc, double varv[],
                int bins) {
#ifdef LOUD_AND_PROUD
    std::cout << LIB_NAME << "BIN TABLE" << std::endl;
    std::cout << LIB_NAME << "name: " << name << std::endl;
    std::cout << LIB_NAME << "varc: " << varc << std::endl;
    for (int i = 0; i<varc; ++i) {
        std::cout << LIB_NAME << "varv[" << i << "]: " << varv[i] << std::endl;
    }
#endif
    double value = 0.0;
    for (int i = 0; i<varc; ++i) {
        value += 10.0*varv[i]*varv[i];
    }
    return value;
}
