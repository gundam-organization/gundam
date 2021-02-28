#include <utility>


#include "XsecDial.hh"
#include "Logger.h"


XsecDial::XsecDial(std::string dial_name)
    : m_name(std::move(dial_name))
{
    _isNormalizationDial_ = false;
    _isSplinesInTTree_    = false;
    Logger::setUserHeaderStr("[XsecDial]");
}

XsecDial::XsecDial(std::string  dial_name, const std::string& fname_binning,
                   const std::string& fname_splines)
    : m_name(std::move(dial_name))
{
    _isNormalizationDial_ = false;
    _isSplinesInTTree_    = false;
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

void XsecDial::SetApplyOnlyOnMap(const std::map<std::string, std::vector<int>>& applyOnlyOnMap_){
    _applyOnlyOnMap_ = applyOnlyOnMap_;
}
void XsecDial::SetDontApplyOnMap(const std::map<std::string, std::vector<int>>& dontApplyOnMap_){
    _dontApplyOnMap_ = dontApplyOnMap_;
}
void XsecDial::SetApplyCondition(std::string applyCondition_){
    _applyCondition_ = std::move(applyCondition_);

    if(GlobalVariables::getChainList().empty()){
        LogFatal << GET_VAR_NAME_VALUE(GlobalVariables::getChainList().empty()) << std::endl;
        throw std::logic_error(GET_VAR_NAME_VALUE(GlobalVariables::getChainList().empty()));
    }

    for(size_t iThread = 0 ; iThread < GlobalVariables::getChainList().size() ; iThread++){
        _applyConditionFormulaeList_.emplace_back(
            new TTreeFormula(
                Form("apply_conditions_%s_%i", m_name.c_str(), iThread),
                _applyCondition_.c_str(),
                GlobalVariables::getChainList().at(iThread)
            )
        );
        GlobalVariables::getChainList().at(iThread)->SetNotify(_applyConditionFormulaeList_.at(iThread));
    }

}

void XsecDial::ReadSplines(const std::string& fname_splines)
{
    v_splines.clear();

    TFile* file_splines = TFile::Open(fname_splines.c_str(), "READ");
    if(file_splines == nullptr)
    {
        LogError << "Failed to open " << fname_splines << std::endl;
        return;
    }

    if(file_splines->Get("InterpolatedBinnedSplines") != nullptr){
        LogWarning << "Splines in TTrees has been detected." << std::endl;
        _isSplinesInTTree_ = true;
    }

    if(_isSplinesInTTree_){

        _interpolatedBinnedSplinesTTree_ = (TTree*) file_splines->Get("InterpolatedBinnedSplines");
        if(_interpolatedBinnedSplinesTTree_ == nullptr){
            LogFatal << "InterpolatedBinnedSplines TTree could not be found for: " << m_name << std::endl;
            throw std::runtime_error("Missing splines");
        }

        _splineBinBuffer_.splitVarNameList.clear();
        _splineBinBuffer_.splitVarValueList.clear();

        // Searching for split values
        for( int iKey = 0 ; iKey < _interpolatedBinnedSplinesTTree_->GetListOfLeaves()->GetEntries() ; iKey++ ){
            std::string leafName = _interpolatedBinnedSplinesTTree_->GetListOfLeaves()->At(iKey)->GetName();
            if(    leafName != "kinematicBin"
                   and leafName != "spline"
                   and leafName != "graph"
                ){
                v_splitVarNames.emplace_back(leafName);
                _splineBinBuffer_.splitVarNameList.emplace_back(leafName);
            }
        }
        _splineBinBuffer_.splitVarValueList.resize(_splineBinBuffer_.splitVarNameList.size());


        // Hooking to the tree
        _interpolatedBinnedSplinesTTree_->SetBranchAddress("kinematicBin", &_splineBinBuffer_.kinematicBin);
        _interpolatedBinnedSplinesTTree_->SetBranchAddress("spline", &_splineBinBuffer_.splinePtr);
        for(int iSplitVar = 0 ; iSplitVar < int(_splineBinBuffer_.splitVarNameList.size()) ; iSplitVar++){
            _interpolatedBinnedSplinesTTree_->SetBranchAddress(
                _splineBinBuffer_.splitVarNameList[iSplitVar].c_str(),
                &_splineBinBuffer_.splitVarValueList[iSplitVar]);
        }

        // Copying splines in RAM
        auto ramBefore = GenericToolbox::getProcessMemoryUsage();
        for( int iEntry = 0 ; iEntry < _interpolatedBinnedSplinesTTree_->GetEntries() ; iEntry++ ){

            _interpolatedBinnedSplinesTTree_->GetEntry(iEntry);

            _splineBinBuffer_.entry = iEntry;
            _splineBinBuffer_.splinePtr = (TSpline3*)_splineBinBuffer_.splinePtr->Clone();

            // Cache for faster spline indexing
            _splinePtrList_.emplace_back(_splineBinBuffer_.splinePtr);
            _splineBinList_.emplace_back(_splineBinBuffer_);
//            _splineNameList_.emplace_back(_splineBinBuffer_.generateBinName());

            // Cache for faster spline read
            _splineCacheList_.emplace_back(SplineCache());
            _splineCacheList_.back().isBeingEdited = false;
            _splineCacheList_.back().cachedDialValue = std::numeric_limits<double>::quiet_NaN();
            _splineCacheList_.back().cachedDialWeight = std::numeric_limits<double>::quiet_NaN();

//            _dialValueCacheList_.emplace_back(0);
//            _weightValueCacheList_.emplace_back(_splineBinBuffer_.splinePtr->Eval(0));
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
        TKey* key;
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

bool XsecDial::GetIsSplinesInTree() const{
    return this->_isSplinesInTTree_;
}
bool XsecDial::GetIsNormalizationDial() const {
    return this->_isNormalizationDial_;
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

int XsecDial::GetSplineIndex(AnaEvent* eventPtr_, SplineBin* eventSplineBinPtr_){

    if(_isNormalizationDial_){
        LogFatal << "Can't find the spline index on a normalization spline." << std::endl;
        throw std::logic_error("Can't find the spline index on a normalization spline.");
    }

    int outSplineIndex = -1;

    bool createSplineBin = false; // if true, will be deleted before leaving
    if(eventSplineBinPtr_ == nullptr){
        createSplineBin = true;
        eventSplineBinPtr_ = new SplineBin();
    }

    eventSplineBinPtr_->kinematicBin = -1;
    for(size_t iBin = 0; iBin < _binEdgesList_.size(); iBin++){
        if(    eventPtr_->GetRecoD1() >= _binEdgesList_[iBin][0].first
           and eventPtr_->GetRecoD1() <  _binEdgesList_[iBin][0].second
           and eventPtr_->GetRecoD2() >= _binEdgesList_[iBin][1].first
           and eventPtr_->GetRecoD2() <  _binEdgesList_[iBin][1].second
            ) {
            eventSplineBinPtr_->kinematicBin = iBin;
            break;
        }
    }

    if(eventSplineBinPtr_->kinematicBin != -1){

        bool isCandidate;
        for(size_t iSpline = 0 ; iSpline < _splineBinList_.size() ; iSpline++){

            isCandidate = true;

            if(_splineBinList_[iSpline].kinematicBin != eventSplineBinPtr_->kinematicBin) continue; // next spline

            // first make the selection on split vars
            for(size_t iSplitVar = 0 ; iSplitVar < v_splitVarNames.size() ; iSplitVar++ ){
                if( _splineBinList_[iSpline].splitVarValueList[iSplitVar] != eventPtr_->GetEventVarInt(v_splitVarNames[iSplitVar]) ){
                    isCandidate = false;
                    break; // next spline
                }
            }

            if(isCandidate){
                outSplineIndex = iSpline;
                break; // LEAVE THE SPLINE LOOP
            }

        }
    }

    if(createSplineBin) delete eventSplineBinPtr_;

    return outSplineIndex;
}

int XsecDial::GetSplineIndex(SplineBin& splineBinToLookFor_){

    // find the bin by generating the name string (slow? -> no FASTER!)
    return GenericToolbox::findElementIndex(splineBinToLookFor_.generateBinName(), _splineNameList_);

}

double XsecDial::GetSplineValue(int index, double dial_value) const
{
    if(index >= 0)
        return v_splines.at(index).Eval(dial_value);
    else
        return 1.0;
}

double XsecDial::GetBoundedValue(int splineIndex_, double dialValue_){

    // Output default
    double dialWeight = 1;

    // Switch between different dial styles:
    if(_isNormalizationDial_){
        // Normalization spline
        return dialValue_;
    }
    else if(_isSplinesInTTree_){
        // Splines in TTrees

        if(_splinePtrList_.empty()){
            // if no splines are stored, there is a problem
            LogFatal << "Can't eval weight. Splines are missing: " << GetName() << std::endl;
            throw std::runtime_error("Missing splines");
        }
        else if(splineIndex_ < 0){
            // The index is invalid, the event is supposed to not be affected
            // returning default weight
            return dialWeight;
        }

        SplineCache* splineCachePtr = &_splineCacheList_[splineIndex_]; // .at() or [] are slower

        // wait for the cache to be filled if another thread is taking care of it
        while(splineCachePtr->isBeingEdited);

        // This if scope might not be thread safe... If corruption happen, set "isBeingEdited" before
        if(splineCachePtr->cachedDialValue == dialValue_
//           and false // disable cache for debug
           ){
            // If the spline has already been calculated, return the cached value
            return splineCachePtr->cachedDialWeight;
        }

        // Take the lock already to avoid another thread to update the cache while it's being computed
        splineCachePtr->isBeingEdited = true;

        TSpline3* currentSplinePtr = _splinePtrList_[splineIndex_]; // .at() or [] are slower

        // Lower bound
        if     (dialValue_ < std::max(m_limit_lo, currentSplinePtr->GetXmin()) ){
            dialValue_ = std::max(m_limit_lo, currentSplinePtr->GetXmin());
        }
        // Higher bound
        else if(dialValue_ > std::min(m_limit_hi, currentSplinePtr->GetXmax()) ){
            dialValue_ = std::min(m_limit_hi, currentSplinePtr->GetXmax());
        }
//            GlobalVariables::getThreadMutex().lock(); // is Eval thread safe?
        dialWeight = currentSplinePtr->Eval(dialValue_);
//            GlobalVariables::getThreadMutex().unlock();

        splineCachePtr->cachedDialValue = dialValue_;
        splineCachePtr->cachedDialWeight = dialWeight;
        splineCachePtr->isBeingEdited = false;

    }
    else if(not v_splines.empty()){
        // Legacy way
        if(splineIndex_ >= 0){
            if(dialValue_ < m_limit_lo)
                dialWeight = v_splines.at(splineIndex_).Eval(m_limit_lo);
            else if(dialValue_ > m_limit_hi)
                dialWeight = v_splines.at(splineIndex_).Eval(m_limit_hi);
            else
                dialWeight = v_splines.at(splineIndex_).Eval(dialValue_);
        }
    }
    else{
        dialWeight += dialValue_; // legacy?
        LogFatal << GET_VAR_NAME_VALUE(this->GetName()) << std::endl;
        LogFatal << "should not be there." << std::endl;
        throw std::logic_error("should not be there.");
    }

    return dialWeight;
}

TSpline3* XsecDial::getCorrespondingSpline(AnaEvent* anaEvent_){

    TSpline3* outSplinePtr = nullptr;

    // need to recreate the holder since this function is call in multithread
    SplineBin currentSplineBin = _splineBinBuffer_;

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
    if(currentSplineBin.kinematicBin != -1){
        // Loop over the defined split variables
        for(int iSplitVar = 0 ; iSplitVar < int(_splineBinBuffer_.splitVarNameList.size()) ; iSplitVar++){
            currentSplineBin.splitVarValueList[iSplitVar] = anaEvent_->GetEventVarInt(_splineBinBuffer_.splitVarNameList[iSplitVar]);
        }
        // Now look for the spline ptr:
        int splineIndex = GetSplineIndex(currentSplineBin);
        if(splineIndex != -1) outSplinePtr = _splineBinList_[splineIndex].splinePtr;
    }

    return outSplinePtr;
}

std::string XsecDial::GetSplineName(int index) const
{
    return std::string(v_splines.at(index).GetTitle());
}
const std::string& XsecDial::GetApplyCondition() const { return _applyCondition_; }

void XsecDial::SetVars(double nominal, double step, double limit_lo, double limit_hi)
{
    m_nominalDial = nominal;
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
    LogWarning << "Name: " <<  m_name << std::endl
             << "Nominal Dial Value: " << m_nominalDial << std::endl
             << "Prior: " << m_prior << std::endl
             << "Step: " << m_step << std::endl
             << "Limits: [" << m_limit_lo << "," << m_limit_hi << "]" << std::endl
             << "Is Normalization Dial: " << (_isNormalizationDial_? "true": "false") << std::endl;

    if(_isNormalizationDial_){
        std::function<std::string(int)> intToString = [](int elementInt_){return std::to_string(elementInt_);};
        if(not _applyOnlyOnMap_.empty())
        {
            LogWarning << "ApplyOnlyOn: {";
            for(const auto& applyCond : _applyOnlyOnMap_){
                LogWarning << " \"" << applyCond.first << "\": [";
                LogWarning << GenericToolbox::joinVectorString(
                    GenericToolbox::convertVectorType(applyCond.second, intToString), ", ");
                LogWarning << "], ";
            }
            LogWarning << "}" << std::endl;
        }
        if(not _dontApplyOnMap_.empty()){
            LogWarning << "DontApplyOn: {";
            for(const auto& applyCond : _dontApplyOnMap_){
                LogWarning << " \"" << applyCond.first << "\": [";
                LogWarning << GenericToolbox::joinVectorString(
                    GenericToolbox::convertVectorType(applyCond.second, intToString), ", ");
                LogWarning << "], ";
            }
            LogWarning << "}" << std::endl;
        }
    }
    else{
        if(_isSplinesInTTree_){
            LogWarning << "Nb of Splines: " << _splinePtrList_.size() << std::endl;
        }
        else{
            LogWarning << "Nb of Splines: " << v_splines.size() << std::endl;
        }
    }



    if(not m_dimensions.empty()){
        LogWarning << "Dimensions:";
        for(const auto& dim : m_dimensions)
            LogWarning << " " << dim;
        LogWarning << std::endl;
    }

    if(print_bins){
        bin_manager.Print();
    }


}

void XsecDial::SetNominal(double nominal) { m_nominalDial = nominal; }
void XsecDial::SetPrior(double prior) { m_prior = prior; }
void XsecDial::SetStep(double step) { m_step = step; }
void XsecDial::SetLimitLo(double limit_lo) { m_limit_lo = limit_lo; }
void XsecDial::SetLimitHi(double limit_hi) { m_limit_hi = limit_hi; }
void XsecDial::SetIsNormalizationDial(bool isNormalizationDial_) { _isNormalizationDial_ = isNormalizationDial_; }
std::vector<TTreeFormula*>& XsecDial::getApplyConditionFormulaeList() {
    return _applyConditionFormulaeList_;
}
