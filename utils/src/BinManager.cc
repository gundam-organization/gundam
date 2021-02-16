#include "BinManager.hh"
#include "Logger.h"
#include "GenericToolbox.h"

BinManager::BinManager()
    : dimension(0), nbins(0)
{
    Logger::setUserHeaderStr("[BinManager]");
}

BinManager::BinManager(const std::string& filename, bool UseNutypeBeammode)
    : dimension(0), nbins(0), fname_binning(filename)
{
    Logger::setUserHeaderStr("[BinManager]");
    SetBinning(fname_binning, UseNutypeBeammode);
}

BinManager::BinManager(const BinManager& source_){

    nbins = source_.nbins;
    dimension = source_.dimension;
    fname_binning = source_.fname_binning;

    for(const auto& bin_edge_row: source_.bin_edges){
        bin_edges.emplace_back(std::vector<std::pair<double, double>>());
        for(const auto& bin_edge_col: bin_edge_row){
            bin_edges.back().emplace_back(std::pair<double, double>(bin_edge_col.first,bin_edge_col.second));
        }
    }

    for(const auto& bin_row: source_.bin_nutype){
        bin_nutype.emplace_back(std::vector<int>());
        for(const auto& bin_col: bin_row){
            bin_nutype.back().emplace_back(bin_col);
        }
    }

    for(const auto& bin_row: source_.bin_beammode){
        bin_beammode.emplace_back(std::vector<int>());
        for(const auto& bin_col: bin_row){
            bin_beammode.back().emplace_back(bin_col);
        }
    }

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

            if(dim == -1){
                dim = GenericToolbox::splitString(line, " ").size()/2;
            }

            // Add a new vector of number pairs for each binning dimension to bin_edges:
            for(int d = 0; d < dim; ++d)
                bin_edges.emplace_back(std::vector<std::pair<double, double>>());

            // Loop over all remaining lines of the binning file:
            while(std::getline(fin, line))
            {
                line = GenericToolbox::trimString(line, " ");
                if(line.empty()) continue;

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

    if(bin_edges.empty()){
        LogFatal << "No bin has been defined. Please check the input file: " << filename << std::endl;
        throw std::logic_error("No bin has been defined.");
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
//    std::cerr << "dimension = " << dimension << std::endl;
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

        if(flag)
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

unsigned int BinManager::GetDimension() const{
    return dimension;
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

void BinManager::MergeBins(unsigned int groupSize_, int dimIndexToRebin_){

    if(not bin_nutype.empty()){
        std::cout << "Error: BinManager::MergeBins only available without UseNutypeBeammode at the moment." << std::endl;
        throw std::logic_error("BinManager::MergeBins only available without UseNutypeBeammode at the moment.");
    }
    if(bin_edges.empty() or groupSize_ == 1) return;


    if(dimIndexToRebin_ == -1){
        for(size_t iDim = 0 ; iDim < bin_edges.size() ; iDim++){
            this->MergeBins(groupSize_, iDim);
        }
        return;
    }

//    return; // DO NOTHING AT THE MOMENT

    // list of configurations where other bins = cte
    std::vector<int> groupBins; // groupBins[iBin] = iGroup
    groupBins.resize(bin_edges[dimIndexToRebin_].size());

    std::vector<std::vector<std::pair<double,double>>> otherBinEdgesList; // otherBinEdges[iGroup][iDim] = bin edge

    //
    for(size_t iBin = 0 ; iBin < bin_edges[dimIndexToRebin_].size() ; iBin++){
        size_t iGroup;
        std::vector<std::pair<double,double>> otherBinEdges;
        for(size_t iDim = 0 ; iDim < bin_edges.size() ; iDim++){
            if(dimIndexToRebin_ == iDim) continue;
            otherBinEdges.emplace_back(std::pair<double, double>(bin_edges[iDim][iBin].first,bin_edges[iDim][iBin].second));
        }

        if(GenericToolbox::doesElementIsInVector(otherBinEdges,otherBinEdgesList)){
            iGroup = GenericToolbox::findElementIndex(otherBinEdges,otherBinEdgesList);
        }
        else{
            otherBinEdgesList.emplace_back(otherBinEdges);
            iGroup = otherBinEdgesList.size()-1;
        }

        groupBins[iBin] = iGroup;
    }

    std::vector<std::vector<int>> groupBinsList; // groupBinsList[iGroup] = {bins indexes}
    groupBinsList.resize(otherBinEdgesList.size());
    for(size_t iGroup = 0 ; iGroup < groupBinsList.size() ; iGroup++){
        for(size_t iBin = 0 ; iBin < bin_edges[dimIndexToRebin_].size() ; iBin++){
            if(groupBins[iBin] == iGroup){
                groupBinsList[iGroup].emplace_back(iBin);
            }
        }
    }



    std::vector<std::vector<std::pair<double, double>>> new_bin_edges;
    new_bin_edges.resize(dimension);
    for(size_t iGroup = 0 ; iGroup < groupBinsList.size() ; iGroup++){

        double lowVal = -1;
        double highVal = -1;
        bool releaseEntry = false;
        for(size_t iBin = 0 ; iBin < groupBinsList[iGroup].size() ; iBin++){

            if(releaseEntry){
                lowVal = highVal;
            }

            releaseEntry = false;

            if(iBin == 0){
                lowVal = bin_edges[dimIndexToRebin_][groupBinsList[iGroup][iBin]].first;
            }
            else if(iBin % groupSize_ == 0 or iBin == groupBinsList[iGroup].size()-1){
                highVal = bin_edges[dimIndexToRebin_][groupBinsList[iGroup][iBin]].second;
                releaseEntry = true;
            }

            if(releaseEntry){
                for(size_t iDim = 0 ; iDim < dimension ; iDim++){
                    if(iDim != dimIndexToRebin_){
                        // every group bins share the same bounds for other dim -> index [0] is chosen
                        new_bin_edges[iDim].emplace_back(
                            std::pair<double,double>(
                                bin_edges[iDim][groupBinsList[iGroup][0]].first,
                                bin_edges[iDim][groupBinsList[iGroup][0]].second
                            )
                        );
                    }
                    else{
                        new_bin_edges[iDim].emplace_back(
                            std::pair<double,double>(
                                lowVal,
                                highVal
                            )
                        );
                    }
                }
            }

        }
    }



//    // list of bins (which will be merged) for each configuration of other bin
//    std::vector<std::vector<int>> binLineIndexList(binningLineList.size());
//    for(size_t iLine = 0; iLine < binLineIndexList.size() ; iLine++){
//        for(size_t iBin = 0 ; iBin < bin_edges[0].size() ; iBin++){
//
//
//
//            for(size_t iDim = 0 ; iDim < bin_edges.size() ; iDim++){
//
//                binningLineList[iLine][iBon]
//
//                if(dimIndexToRebin_ == iDim) continue;
//                otherBinIndexList.emplace_back(iDim);
//            }
//        }
//    }
//
//
//    std::map<std::vector<int>, std::vector<int>> binningLine; // [list of other bins index] = {list of bin which share the same reference}
//
//    for(size_t iDim = 0 ; iDim < bin_edges.size() ; iDim++){
//        new_bin_edges[iDim].resize(bin_edges[iDim].size()/ groupSize_ +1);
//        size_t iReBin = 0;
//        for(size_t iBin = 0 ; iBin < bin_edges[iDim].size() ; iBin++){
//
//            if(iBin == 0){
//                // open the first bin
//                new_bin_edges[iDim][iReBin].first = bin_edges[iDim][iBin].first;
//            }
//            else if(iBin == bin_edges[iDim].size()-1){
//                // close the last bin
//                new_bin_edges[iDim][iReBin].second = bin_edges[iDim][iBin].second;
//            }
//            else if( iBin % groupSize_ == 0 ){
//                // close the previous bin
//                new_bin_edges[iDim][iReBin].second = bin_edges[iDim][iBin].second;
//
//                // open the next
//                iReBin++;
//                new_bin_edges[iDim][iReBin].first = bin_edges[iDim][iBin].first;
//            }
//
//        }
//    }

    // update members
    bin_edges = new_bin_edges;
    nbins = bin_edges.at(0).size();

}