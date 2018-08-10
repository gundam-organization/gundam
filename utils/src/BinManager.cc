#include "BinManager.hh"

BinManager::BinManager()
    : dimension(0), nbins(0)
{
}

BinManager::BinManager(const std::string& filename)
    : dimension(0), nbins(0), fname_binning(filename)
{
    SetBinning(fname_binning);
}

int BinManager::SetBinning(const std::string& filename)
{
    double dim = 0;
    std::ifstream fin(filename, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: Failed to open " << filename << std::endl;
        return 0;
    }
    else
    {
        std::string line;
        if(std::getline(fin, line))
        {
            std::stringstream ss(line);
            ss >> dim;
        }

        for(int d = 0; d < dim; ++d)
            bin_edges.emplace_back(std::vector<std::pair<double, double>>());

        while(std::getline(fin, line))
        {
            std::stringstream ss(line);
            double bin_low, bin_high;

            for(int d = 0; d < dim; ++d)
            {
                if(!(ss >> bin_low >> bin_high))
                {
                    std::cerr << "[ERROR]: Bad line format: " << std::endl
                              << line << std::endl;
                    continue;
                }

                bin_edges.at(d).emplace_back(std::make_pair(bin_low, bin_high));
            }
        }
        fin.close();
    }

    nbins = bin_edges.at(0).size();
    dimension = dim;

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
            return i;
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

bool BinManager::CheckBinIndex(const int i, const int d, const double val) const
{
    return bin_edges[d][i].first <= val && val < bin_edges[d][i].second;
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
