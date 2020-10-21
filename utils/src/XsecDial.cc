#include "XsecDial.hh"
#include "Logger.h"


XsecDial::XsecDial(const std::string& dial_name)
    : m_name(dial_name)
{
    _useSplineSplitMapping_ = false;
    Logger::setUserHeaderStr("[XsecDial]");
}

XsecDial::XsecDial(const std::string& dial_name, const std::string& fname_binning,
                   const std::string& fname_splines)
    : m_name(dial_name)
{
    _useSplineSplitMapping_ = false;
    Logger::setUserHeaderStr("[XsecDial]");
    SetBinning(fname_binning);
    ReadSplines(fname_splines);
}

void XsecDial::SetBinning(const std::string& fname_binning)
{
    if(not fname_binning.empty()){
        bin_manager.SetBinning(fname_binning);
        nbins = bin_manager.GetNbins();
        _binEdgesList_ = bin_manager.GetEdgeVector();
    }
}

void XsecDial::ReadSplines(const std::string& fname_splines)
{
    v_splines.clear();

    if(fname_splines.empty()){
        LogWarning << "No spline file specified for " << m_name << std::endl;
        LogWarning << " > Will be treated as normalisation factor." << std::endl;
        return;
    }

    TFile* file_splines = TFile::Open(fname_splines.c_str(), "READ");
    if(file_splines == nullptr)
    {
        LogError << "Failed to open " << fname_splines << std::endl;
        return;
    }

    if(file_splines->Get("UnbinnedSplines") != nullptr){
        LogWarning << "Splines in TTrees has been detected." << std::endl;
        _useSplineSplitMapping_ = true;
    }

    if(_useSplineSplitMapping_){

        _interpolatedBinnedSplinesTTree_ = (TTree*) file_splines->Get("InterpolatedBinnedSplines");
        if(_interpolatedBinnedSplinesTTree_ == nullptr){
            LogError << "No splines has been found for " << m_name << std::endl;
            return;
        }

        // Hooking to the tree
        _interpolatedBinnedSplinesTTree_->SetBranchAddress("kinematicBin", &_splineSplitBinHandler_.kinematicBin);
        _interpolatedBinnedSplinesTTree_->SetBranchAddress("spline", &_splineSplitBinHandler_.splineHandler);
//        _interpolatedBinnedSplinesTTree_->SetBranchAddress("graph", &_splineSplitBinHandler_.graphHandler);
        for( int iKey = 0 ; iKey < _interpolatedBinnedSplinesTTree_->GetListOfLeaves()->GetEntries() ; iKey++ ){
            std::string leafName = _interpolatedBinnedSplinesTTree_->GetListOfLeaves()->At(iKey)->GetName();
            if(    leafName != "kinematicBin"
               and leafName != "spline"
               and leafName != "graph"
               ){
                v_splitVarNames.emplace_back(leafName);
                _splineSplitBinHandler_.splitVarValue[leafName] = -1; // allocate memory
                _interpolatedBinnedSplinesTTree_->SetBranchAddress(leafName.c_str(), &_splineSplitBinHandler_.splitVarValue[leafName]);
            }
        }

        // Copying splines in RAM
        auto ramBefore = GenericToolbox::getProcessMemoryUsage();
        TSpline3* splineTemp = nullptr;
        for( int iEntry = 0 ; iEntry < _interpolatedBinnedSplinesTTree_->GetEntries() ; iEntry++ ){
            _interpolatedBinnedSplinesTTree_->GetEntry(iEntry);
            splineTemp = (TSpline3*) _splineSplitBinHandler_.splineHandler->Clone();
//            v_splines.emplace_back(*splineTemp);
//            _splineMapping_[_splineSplitBinHandler_.getBinName()] = splineTemp;
            v_splinesPtr.emplace_back(splineTemp);
            v_splineNames.emplace_back(_splineSplitBinHandler_.getBinName());
        }

        file_splines -> Close();
        delete file_splines;
        auto ramAfter = GenericToolbox::getProcessMemoryUsage();
        LogDebug << "The splines are taking "
                 << GenericToolbox::parseSizeUnits(ramAfter - ramBefore)
                 << " of RAM." << std::endl;

    }
    else{
        TIter key_list(file_splines -> GetListOfKeys());
        TKey* key = nullptr;
        while((key = (TKey*)key_list.Next()))
        {
            if (strcmp(key->ReadObj()->GetName(), "Graph") == 0){
                auto* graph = (TGraph*)key -> ReadObj();
                v_splines.emplace_back(TSpline3(graph->GetName(), graph));
                //v_splines.emplace_back(*spline);
            }
        }
        file_splines -> Close();
        delete file_splines;

        v_splines.shrink_to_fit();
    }

}

bool XsecDial::GetUseSplineSplitMapping() const{
    return this->_useSplineSplitMapping_;
}

int XsecDial::GetSplineIndex(const std::vector<int>& var, const std::vector<double>& bin) const
{
    if(var.size() != m_dimensions.size())
        return -1;

    int idx = bin_manager.GetBinIndex(bin);
    for(int i = 0; i < var.size(); ++i)
        idx += var[i] * m_dimensions[i];
    return idx;
}

int XsecDial::GetSplineIndex(AnaEvent* anaEvent_)
{
    int outSplineIndex = -1;

    if(not v_splineNames.empty())
    {
        // need to recreate the holder since this function is call in multithread
        SplineBin* currentSplineBin;

//        currentSplineBin = &_splineSplitBinHandler_;
        currentSplineBin = new SplineBin();

        currentSplineBin->D1Reco = anaEvent_->GetRecoD1();
        currentSplineBin->D2Reco = anaEvent_->GetRecoD2();
        // Identify Kinematic Bin
        currentSplineBin->kinematicBin = -1;
        for(int iBin = 0; iBin < int(_binEdgesList_.size()); iBin++)
        {
            if(currentSplineBin->D1Reco >= _binEdgesList_[iBin][0].first
               and currentSplineBin->D1Reco < _binEdgesList_[iBin][0].second
               and currentSplineBin->D2Reco >= _binEdgesList_[iBin][1].first
               and currentSplineBin->D2Reco < _binEdgesList_[iBin][1].second)
            {
                currentSplineBin->kinematicBin = iBin;
                break;
            }
        }
        if(currentSplineBin->kinematicBin != -1)
        {
            // Loop over the defined split variables
            for(const auto& splineVarName : v_splitVarNames)
            {
                currentSplineBin->splitVarValue[splineVarName]
                    = anaEvent_->GetEventVarInt(splineVarName);
            }

            //
            outSplineIndex = GenericToolbox::findElementIndex(currentSplineBin->getBinName(), v_splineNames);
        }

        delete currentSplineBin;
    } // not v_splineNames.empty()

    return outSplineIndex;
}

double XsecDial::GetSplineValue(int index, double dial_value) const
{
    if(index >= 0)
        return v_splines.at(index).Eval(dial_value);
    else
        return 1.0;
}

double XsecDial::GetBoundedValue(int index, double dial_value) const
{
    double dialWeight = 1;

    // Legacy way
    if(not v_splines.empty())
    {
        if(index >= 0)
        {
            if(dial_value < m_limit_lo)
                dialWeight = v_splines.at(index).Eval(m_limit_lo);
            else if(dial_value > m_limit_hi)
                dialWeight = v_splines.at(index).Eval(m_limit_hi);
            else
                dialWeight = v_splines.at(index).Eval(dial_value);
        }
        else
            return 1.0;
    }
    // Splines in TTree
    else if(not v_splinesPtr.empty())
    {
        if(index >= 0)
        {
            if     ( dial_value < std::max(m_limit_lo, v_splinesPtr[index]->GetXmin()) ){
                dial_value = std::max(m_limit_lo, v_splinesPtr[index]->GetXmin());
            }
            else if( dial_value > std::min(m_limit_hi, v_splinesPtr[index]->GetXmax()) ){
                dial_value = std::min(m_limit_hi, v_splinesPtr[index]->GetXmax());
            }
            dialWeight = v_splinesPtr[index]->Eval(dial_value);
        }
    }
    // Norm splines
    else{
        // norm spline
        // TODO: check apply_on variables
        dialWeight = 1 + dial_value;
    }

    return dialWeight;
}

double XsecDial::GetBoundedValue(AnaEvent* anaEvent_, double dial_value_){

    double dialWeight = 1;

    if(_splineMapping_.empty()){
        // norm spline
        // TODO: check apply_on variables
        dialWeight = 1 + dial_value_;
    }
    else{
        TSpline3* selectedSpline = getCorrespondingSpline(anaEvent_);
        if(selectedSpline != nullptr){
            double dialValue;
            if     ( dial_value_ < std::max(m_limit_lo, selectedSpline->GetXmin()) ){
                dialValue = std::max(m_limit_lo, selectedSpline->GetXmin());
            }
            else if( dial_value_ > std::min(m_limit_hi, selectedSpline->GetXmax()) ){
                dialValue = std::min(m_limit_hi, selectedSpline->GetXmax());
            }
            else{
                dialValue = dial_value_;
            }
            dialWeight = selectedSpline->Eval(dialValue);
            if( dialWeight == 0 ){
                LogError << "0 weight detected: " << this->GetName() << ", spline(" << selectedSpline << ")->Eval(" << dialValue << ") = " << dialWeight << std::endl;
            }
        }
    }

    return dialWeight;

}

TSpline3* XsecDial::getCorrespondingSpline(AnaEvent* anaEvent_){

    if(_splineMapping_.empty()){
        return nullptr;
    }

    if(
//        true
        not _eventSplineMapping_[anaEvent_].first
        ){

        // need to recreate the holder since this function is call in multithread
        SplineBin currentSplineBin;

        // reset all vars to -1
        currentSplineBin.reset();

        currentSplineBin.D1Reco = anaEvent_->GetRecoD1();
        currentSplineBin.D2Reco = anaEvent_->GetRecoD2();
        // Identify Kinematic Bin
        currentSplineBin.kinematicBin = -1;
        for(int iBin = 0 ; iBin < int(_binEdgesList_.size()) ; iBin++){
            if(     currentSplineBin.D1Reco >= _binEdgesList_[iBin][0].first
                    and currentSplineBin.D1Reco <  _binEdgesList_[iBin][0].second
                    and currentSplineBin.D2Reco >= _binEdgesList_[iBin][1].first
                    and currentSplineBin.D2Reco <  _binEdgesList_[iBin][1].second
                ){
                currentSplineBin.kinematicBin = iBin;
                break;
            }
        }
        if(currentSplineBin.kinematicBin == -1){
//            return nullptr;
            #pragma omp critical
            {
                _eventSplineMapping_[anaEvent_].second = nullptr;
                _eventSplineMapping_[anaEvent_].first = true;
            }
        }
        else{
            // Loop over the defined split variables
            for(auto& splineSplitBinPair : _splineSplitBinHandler_.splitVarValue){
                currentSplineBin.splitVarValue[splineSplitBinPair.first] = anaEvent_->GetEventVarInt(splineSplitBinPair.first);
            }
            #pragma omp critical
            {
            _eventSplineMapping_[anaEvent_].second = _splineMapping_[_splineSplitBinHandler_.getBinName()];
            _eventSplineMapping_[anaEvent_].first = true;
            }
//            return _splineMapping_[_splineSplitBinHandler_.getBinName()];
        }
    }

    return _eventSplineMapping_[anaEvent_].second;
}

std::string XsecDial::GetSplineName(int index) const
{
    return std::string(v_splines.at(index).GetTitle());
}

void XsecDial::SetVars(double nominal, double step, double limit_lo, double limit_hi)
{
    m_nominal = nominal;
    m_step = step;
    m_limit_lo = limit_lo;
    m_limit_hi = limit_hi;
}

void XsecDial::SetDimensions(const std::vector<int>& dim)
{
    m_dimensions = dim;
}

void XsecDial::Print(bool print_bins) const
{
    LogAlert << "Name: " <<  m_name << std::endl
             << "Nominal: " << m_nominal << std::endl
             << "Step: " << m_step << std::endl
             << "Limits: [" << m_limit_lo << "," << m_limit_hi << "]" << std::endl
             << "Splines: " << v_splines.size() << std::endl;

    if(not m_dimensions.empty()){
        LogAlert << "Dimensions:";
        for(const auto& dim : m_dimensions)
            LogAlert << " " << dim;
        LogAlert << std::endl;
    }

    if(print_bins)
        bin_manager.Print();
}
