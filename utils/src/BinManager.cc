#include "BinManager.hh"

BinManager::BinManager()
    : dimension(0), nbins(0)
{
}

BinManager::BinManager(const std::string& filename, bool UseNutypeBeammode)
    : dimension(0), nbins(0), fname_binning(filename)
{
    SetBinning(fname_binning, UseNutypeBeammode);
}

// Reads in a binning text file with the binning and stores it in bin_edges vector:
int BinManager::SetBinning(const std::string& filename, bool UseNutypeBeammode)
{
    // dim will give the dimensionality of the binning:
    double dim = 0;

    // Open binning file for reading:
    std::ifstream fin(filename, std::ios::in);

    // Throw an error message if file cannot be opened and return 0:
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: Failed to open " << filename << std::endl;
        return 0;
    }

    // If file can be opened, there are 2 possible options depending on whether UseNutypeBeammode was set to true (only for flux binning) or false (default):
    else
    {
        // If UseNutypeBeammode is set to false (default value), the first line of the binning file (should be a single number) gives the dimensionality of the binning and all other lines give the bin edges:
        if(!UseNutypeBeammode)
        {
            // First line of binning file gives dimensionality of the binning (stored in dim):
            std::string line;
            if(std::getline(fin, line))
            {
                std::stringstream ss(line);
                ss >> dim;
            }

            // Add a new vector of number pairs for each binning dimension to bin_edges:
            for(int d = 0; d < dim; ++d)
                bin_edges.emplace_back(std::vector<std::pair<double, double>>());

            // Loop over all remaining lines of the binning file:
            while(std::getline(fin, line))
            {
                // ss holds the content of the current line:
                std::stringstream ss(line);

                // Doubles to hold the values of the bin edges:
                double bin_low, bin_high;

                // Loop over the number of dimensions in the binning:
                for(int d = 0; d < dim; ++d)
                {
                    // If we cannot extract the bin edges from the next pair of numbers, an error message is printed but we continue with the rest of the file:
                    if(!(ss >> bin_low >> bin_high))
                    {
                        std::cerr << "[ERROR]: Bad line format: " << std::endl
                                  << line << std::endl;
                        continue;
                    }

                    // Bin edges are added to the bin_edges vector (at current binning dimension d) as a pair:
                    bin_edges.at(d).emplace_back(std::make_pair(bin_low, bin_high));
                }
            }
        }

        // If UseNutypeBeammode is set to true (should only be the case for flux binning), the first two entries in each line give the bin edges followed by the neutrino type and then by the beam mode (FHC or RHC). The fisrt line should still give the dimensionality which should be 1:
        else
        {
            // First line of binning file gives dimensionality of the binning (stored in dim). This should be 1 for the flux binning:
            std::string line;
            if(std::getline(fin, line))
            {
                std::stringstream ss(line);
                ss >> dim;
            }

            // Add a new vector of number pairs to bin_edges, and a new vector of integers to bin_nutype and bin_beammode:
            bin_edges.emplace_back(std::vector<std::pair<double, double>>());
            bin_nutype.emplace_back(std::vector<int>());
            bin_beammode.emplace_back(std::vector<int>());

            // Loop over all remaining lines of the binning file:
            while(std::getline(fin, line))
            {
                // ss holds the content of the current line:
                std::stringstream ss(line);

                // Doubles to hold the values of the bin edges and integers to hold the neutrino type and the beammode of the current bin (defined in the current line):
                double bin_low, bin_high;
                int nutype, beammode;

                // If we cannot extract the bin edges from this line, an error message is printed but we continue with the rest of the file:
                if(!(ss >> bin_low >> bin_high >> nutype >> beammode))
                {
                    std::cerr << "[ERROR]: Bad line format: " << std::endl
                              << line << std::endl;
                    continue;
                }

                // Bin edges are added to the bin_edges vector as a pair, neutrino type and beammode are added to bin_nutype and bin_beammode (at binning dimension d=0 since we only have 1 binning dimension):
                bin_edges.at(0).emplace_back(std::make_pair(bin_low, bin_high));
                bin_nutype.at(0).emplace_back(nutype);
                bin_beammode.at(0).emplace_back(beammode);
            }
        }

        // Close binning text file:
        fin.close();
    }

    // Store the number of bins and the binning dimensionality:
    nbins = bin_edges.at(0).size();
    dimension = dim;

    // Return the binning dimensionality:
    return dim;
}

int BinManager::GetNbins() const
{
    return nbins;
}

int BinManager::GetBinIndex(const std::vector<double>& val) const
{
    if(val.size() != dimension)
    {
        std::cout << "[ERROR]: Number of parameters does not match dimension!" << std::endl;
        exit(EXIT_FAILURE);
        return -1;
    }

    for(int i = 0; i < nbins; ++i)
    {
        bool flag = true;
        for(int d = 0; d < dimension; ++d)
        {
            flag = flag && CheckBinIndex(i, d, val.at(d));
        }

        if(flag == true)
        {
            return i;
        }
    }

    return -1;
}

int BinManager::GetBinIndex(const std::vector<double>& val, const int val_nutype, const int val_beammode) const
{
    if(val.size() != dimension)
    {
        std::cout << "[ERROR]: Number of parameters does not match dimension!" << std::endl;
        exit(EXIT_FAILURE);
        return -1;
    }

    for(int i = 0; i < nbins; ++i)
    {
        bool flag = true;
        for(int d = 0; d < dimension; ++d)
        {
            flag = flag && CheckBinIndex(i, d, val.at(d), val_nutype, val_beammode);
        }

        if(flag == true)
        {
            return i;
        }
    }

    return -1;
}

std::vector<double> BinManager::GetBinVector(const double d) const
{
    std::vector<double> v;
    for(int i = 0; i < nbins; ++i)
        v.emplace_back(bin_edges.at(d).at(i).first);
    v.emplace_back(bin_edges.at(d).back().second);

    std::sort(v.begin(), v.end());
    auto iter = std::unique(v.begin(), v.end());
    v.erase(iter, v.end());

    return v;
}

double BinManager::GetBinWidth(const int i) const
{
    if(i >= nbins)
    {
        std::cout << "[WARNING]: Index " << i << " out of bounds." << std::endl;
        return 1.0;
    }

    double total_width = 1.0;
    for(int d = 0; d < dimension; ++d)
    {
        double dim_width = std::abs(bin_edges[d][i].second - bin_edges[d][i].first);
        total_width *= dim_width;
    }

    return total_width;
}

double BinManager::GetBinWidth(const int i, const int d) const
{
    return std::abs(bin_edges[d][i].second - bin_edges[d][i].first);
}

bool BinManager::CheckBinIndex(const int i, const int d, const double val) const
{
    return bin_edges[d][i].first <= val && val < bin_edges[d][i].second;
}

bool BinManager::CheckBinIndex(const int i, const int d, const double val, const int val_nutype, const int val_beammode) const
{
    return bin_edges[d][i].first <= val && val < bin_edges[d][i].second && bin_nutype[d][i] == val_nutype && bin_beammode[d][i] == val_beammode;
}

void BinManager::Print() const
{
    std::cout << "Bin Edges: " << std::endl;
    std::cout << std::setprecision(3) << std::fixed;
    for(int i = 0; i < nbins; ++i)
    {
        std::cout << i;
        for(int d = 0; d < dimension; ++d)
        {
            std::cout << std::setw(10) << bin_edges[d][i].first
                      << std::setw(10) << bin_edges[d][i].second;
        }
        std::cout << std::endl;
    }
    std::cout.unsetf(std::ios::fixed);
    std::cout.precision(6);
}
