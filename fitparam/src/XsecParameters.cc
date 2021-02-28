#include "XsecParameters.hh"
#include "GenericToolbox.h"
#include "Logger.h"
#include <future>
#include "GlobalVariables.h"

XsecParameters::XsecParameters(const std::string& name)
{
    m_name = name;
    Npar = 0;
    _enableZeroWeightFenceGate_ = false;
    Logger::setUserHeaderStr("[XsecParameters]");
}

XsecParameters::~XsecParameters() { ; }

void XsecParameters::InitEventMap(std::vector<AnaSample*>& samplesList_, int mode)
{
    LogWarning << "Initializing Event Map..." << std::endl;

    InitParameters();

    if(Npar == 0)
    {
        LogError << "In XsecParameters::InitEventMap\n"
                 << "No parameters delcared. Not building event map."
                 << std::endl;
    }

    _eventDialSplineIndexList_.clear();

    LogDebug << "Claiming memory for event mapping..." << std::endl;
    auto ramBefore = GenericToolbox::getProcessMemoryUsage();

    _eventDialSplineIndexList_.resize(samplesList_.size());
    for(std::size_t iSample = 0; iSample < samplesList_.size(); ++iSample){

        _eventDialSplineIndexList_.at(iSample).resize(samplesList_.at(iSample)->GetN());

        // Count the number of spline-dials that need to be cached and indexed
        int detectorIndex = XsecParameters::GetDetectorIndex(samplesList_.at(iSample)->GetDetector());

        std::vector<std::vector<int>> sample_map(samplesList_[iSample] -> GetN());
        for(int iEvent = 0; iEvent < samplesList_[iSample] -> GetN(); iEvent++){
            _eventDialSplineIndexList_.at(iSample).at(iEvent).resize(_dialsList_.at(detectorIndex).size());
        }
    }
    LogDebug << GenericToolbox::parseSizeUnits(GenericToolbox::getProcessMemoryUsage() - ramBefore)
             << " of RAM has been taken." << std::endl;

    int totalNbEvents = 0.;
    for(const auto& anaSample : samplesList_){
        totalNbEvents += anaSample->GetN();
    }

    auto* counterPtr = new int();
    std::function<void(int)> fillEventMapThread = [counterPtr, samplesList_, mode, this](int iThread_){

        bool isMultiThreaded = (iThread_ != -1);
        if(not isMultiThreaded) iThread_ = 0; // for the progress bar

        auto* splineBinBufferPtr = new SplineBin();

        int splineIndex;
        std::string progressBarTitle;
        // Speed optimization: avoid trying to fetch each sub index in vector
        AnaEvent* anaEventPtr;
        AnaSample* anaSamplePtr;
        XsecDial* currentDial;
        std::vector<int>* eventDialsIndexListPtr;
        std::vector<XsecDial>* detectorDialsList;
        std::vector<std::vector<int>>* sampleDialsIndexListPtr;


        for(std::size_t iSample = 0; iSample < samplesList_.size(); ++iSample)
        {
            if(not isMultiThreaded){
                LogInfo << "Mapping events in samplesList_: " << samplesList_[iSample]->GetName()
                        << " (" << samplesList_[iSample] -> GetN() << " events)" << std::endl;
                progressBarTitle = LogWarning.getPrefixString() + "Associating events with splines";
            }

            anaSamplePtr = samplesList_.at(iSample);
            sampleDialsIndexListPtr = &_eventDialSplineIndexList_.at(iSample);

            size_t nbMcEvents = anaSamplePtr->GetMcEvents().size();
            for( size_t iEvent = 0 ; iEvent < nbMcEvents ; iEvent++ ){

                if(isMultiThreaded){
                    if( iEvent % GlobalVariables::getNbThreads() != iThread_ ){
                        continue; // skip
                    }
                }

                if( iThread_ == 0 ){
                    (*counterPtr) += GlobalVariables::getNbThreads();
                }

                eventDialsIndexListPtr = &sampleDialsIndexListPtr->at(iEvent);
                anaEventPtr            = anaSamplePtr->GetEvent(iEvent);
                detectorDialsList = &_dialsList_.at(this->GetDetectorIndex(anaSamplePtr->GetDetector()));

                for( size_t iDial = 0 ; iDial < detectorDialsList->size() ; iDial++ ){
                    currentDial = &detectorDialsList->at(iDial);
                    splineIndex = -1; // NO SPLINE = Not affected

                    // If this point is reached, then a valid spline index should be associated to the anaEventPtr
                    if( currentDial->GetIsNormalizationDial() ){
                        bool skipThisDial = false;
                        if(not currentDial->GetApplyOnlyOnMapPtr()->empty()){
                            for(const auto& applyCondition : *currentDial->GetApplyOnlyOnMapPtr()){
                                if(not GenericToolbox::doesElementIsInVector(
                                       anaEventPtr->GetEventVarInt(applyCondition.first),
                                    applyCondition.second)
                                    ){
                                    skipThisDial = true;
                                    break;
                                }
                            }
                        }

                        if(not skipThisDial and not currentDial->GetDontApplyOnMapPtr()->empty()){
                            for(const auto& applyCondition : *currentDial->GetDontApplyOnMapPtr()){
                                if( GenericToolbox::doesElementIsInVector(
                                       anaEventPtr->GetEventVarInt(applyCondition.first),
                                    applyCondition.second
                                )){
                                    skipThisDial = true;
                                    break;
                                }
                            }
                        }

                        if(not skipThisDial and not currentDial->GetApplyCondition().empty()){
                            if(GlobalVariables::getChainList().empty()){
                                LogFatal << GET_VAR_NAME_VALUE(GlobalVariables::getChainList().empty());
                                LogFatal << ": Can't check cut condition." << std::endl;
                                throw std::logic_error(GET_VAR_NAME_VALUE(GlobalVariables::getChainList().empty()));
                            }
                            GlobalVariables::getChainList().at(iThread_)->GetEntry(
                                anaEventPtr->GetEvId());
//                            GlobalVariables::getMcTreePtr()->GetEntry(anaEventPtr->GetEvId());
                            for(int jInstance = 0; jInstance < currentDial->getApplyConditionFormulaeList().at(iThread_)->GetNdata(); jInstance++) {
                                if ( currentDial->getApplyConditionFormulaeList().at(iThread_)->EvalInstance(jInstance) == 0 ) {
                                    skipThisDial = true;
                                    break;
                                }
                            }
                        }

                        if(not skipThisDial){
                            splineIndex = 0; // event will be renormalized
                        }
                    }
                    else if( currentDial->GetIsSplinesInTree() ){
                        splineIndex = currentDial->GetSplineIndex(anaEventPtr, splineBinBufferPtr);
                    }
                    else{
                        // OLD ROUTINE
                        splineIndex
                            = currentDial->GetSplineIndex(
                            std::vector<int>    {anaEventPtr->GetSampleType(),
                                             anaEventPtr->GetReaction()},
                            std::vector<double> {anaEventPtr->GetTrueD2(),
                                                anaEventPtr->GetTrueD1()}
                        );

                        if(splineIndex == BADBIN){
                            LogWarning << "Event falls outside spline range.\n"
                                       << "This anaEventPtr will be ignored in the analysis."
                                       << std::endl;
                            anaEventPtr->AddEvWght(0.0);
                        }

                        if(mode == 1 && anaEventPtr->isSignalEvent())
                            splineIndex = PASSEVENT;
                    }

                    eventDialsIndexListPtr->at(iDial) = splineIndex;

                } // iDial

            } // iEvent

        } // iSample

        delete splineBinBufferPtr;
    };

    LogInfo << "Associating events with spline indexes..." << std::endl;
    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(0);
    if(GlobalVariables::getNbThreads() > 1){
        std::vector<std::future<void>> threadsList;
        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++ ){
            threadsList.emplace_back(
                std::async( std::launch::async, std::bind(fillEventMapThread, iThread) )
            );
        }

        std::string progressBarPrefix = LogWarning.getPrefixString() + "Associating events with splines";
        GenericToolbox::ProgressBar::lastDisplayedPercentValue = -1;
        for( int iThread = 0 ; iThread < GlobalVariables::getNbThreads(); iThread++ ){
            while(threadsList[iThread].wait_for(std::chrono::milliseconds(33)) != std::future_status::ready){
                GenericToolbox::displayProgressBar(*counterPtr, totalNbEvents, progressBarPrefix);
            }
            threadsList[iThread].get();
        }
    }
    else{
        fillEventMapThread(-1);
    }
    LogDebug << "Event-splines assotiation took: "
             << GenericToolbox::parseTimeUnit(GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(0))
             << std::endl;

    LogInfo << "Event mapping has been done." << std::endl;

}

void XsecParameters::InitParameters()
{
    unsigned int offset = 0;
    for(size_t detectorIndex = 0 ; detectorIndex < v_detectors.size() ; detectorIndex++){
        v_offsets.emplace_back(offset);
        for(const auto& xsecDial : _dialsList_.at(detectorIndex))
        {
            pars_name.emplace_back(Form("%s_%s", v_detectors[detectorIndex].c_str(), xsecDial.GetName().c_str()));
            pars_prior.push_back(xsecDial.GetPrior());
            pars_step.push_back(xsecDial.GetStep());
            pars_limlow.push_back(xsecDial.GetLimitLow());
            pars_limhigh.push_back(xsecDial.GetLimitHigh());
            pars_fixed.push_back(false);

            _dialCacheWeight_.emplace_back();
            _dialCacheWeight_.back().isBeingEdited = false;
            _dialCacheWeight_.back().fitParameterValue = std::numeric_limits<double>::quiet_NaN();
            _dialCacheWeight_.back().cachedWeight = std::numeric_limits<double>::quiet_NaN();

            LogInfo << "Added " << v_detectors[detectorIndex] << "_" << xsecDial.GetName()
                    << std::endl;
        }
        LogInfo << "Total " << _dialsList_.at(detectorIndex).size() << " parameters at "
                << offset << " for " << v_detectors[detectorIndex] << std::endl;

        offset += _dialsList_.at(detectorIndex).size();
    }

    Npar = pars_name.size();
    pars_original = pars_prior;

    if(m_decompose) {
        LogDebug << "Decomposing covariance matrix..." << std::endl;
        pars_prior = eigen_decomp -> GetDecompParameters(pars_prior);
        pars_limlow = std::vector<double>(Npar, -100);
        pars_limhigh = std::vector<double>(Npar, 100);

        const int idx = eigen_decomp -> GetInfoFraction(m_info_frac);
        for(int i = idx; i < Npar; ++i)
            pars_fixed[i] = true;

        LogInfo << "Decomposed parameters.\n"
                << "Keeping the " << idx << " largest eigen values.\n"
                << "Corresponds to " << m_info_frac * 100.0
                << "\% total variance.\n";
    }
}

void XsecParameters::ReWeight(AnaEvent* event, const std::string& detectorName, int nsample, int nevent, std::vector<double>& params){

    if(_eventDialSplineIndexList_.empty()) // need to build an event map first
    {
        LogError << "In XsecParameters::ReWeight()\n"
                 << "Need to build event map index for " << m_name << std::endl;
        return;
    }

    double totalEventWeight = 1;
    int detectorIndex = this->GetDetectorIndex(detectorName);
    auto* detectorDialListPtr       = &_dialsList_.at(detectorIndex);
    auto* eventDialSplineIndexCache = &_eventDialSplineIndexList_.at(nsample).at(nevent);
    double currentDialWeight;
    DialCache* dialCachePtr;

    for( size_t iDial = 0 ; iDial < detectorDialListPtr->size() ; iDial++ ){


        if(eventDialSplineIndexCache->at(iDial) < 0){
            // Skip this event (norm dial are also set to -1 index if they should not be applied)
            continue;
        }

        dialCachePtr = &_dialCacheWeight_.at(iDial + v_offsets.at(detectorIndex));

        // wait if this re-weight value is being edited
        while(dialCachePtr->isBeingEdited);

        if(
            dialCachePtr->fitParameterValue == params.at(iDial + v_offsets.at(detectorIndex))
            and dialCachePtr->cachedWeight == dialCachePtr->cachedWeight // IS NOT A NAN
//            and false // DISABLE CACHE for debug
           ){
            currentDialWeight = dialCachePtr->cachedWeight; // get from cache
        }
        else{

            // make other threads wait
            dialCachePtr->isBeingEdited = true;
            dialCachePtr->fitParameterValue = params.at(iDial + v_offsets.at(detectorIndex));

            // Then the dial has to affect the event (spline or normalization)
            currentDialWeight = detectorDialListPtr->at(iDial).GetBoundedValue(
                eventDialSplineIndexCache->at(iDial), dialCachePtr->fitParameterValue
            );

            // filling cache
            dialCachePtr->cachedWeight = currentDialWeight;
            dialCachePtr->isBeingEdited = false;

        }

        if( currentDialWeight != currentDialWeight // is NaN!!! (cache mess up?)
            or (_enableZeroWeightFenceGate_ and currentDialWeight == 0) ){
            GlobalVariables::getThreadMutex().lock(); // stop all other threads
            Logger::quietLineJump();
            XsecDial* currentDial = &detectorDialListPtr->at(iDial);
            LogFatal << "Dial: " << currentDial->GetName() << " returned invalid weight." << std::endl;
            LogFatal << "Weight: " << GET_VAR_NAME_VALUE(currentDialWeight) << std::endl;
            currentDial->Print();
            LogFatal << "Event: " << std::endl;
            event->Print();
            if(not currentDial->GetIsNormalizationDial()) LogFatal << "Spline Index: " << eventDialSplineIndexCache->at(iDial) << std::endl;
            LogFatal << "Dial Value: " << params.at(iDial + v_offsets.at(detectorIndex)) << std::endl;
            if( dialCachePtr->fitParameterValue == params.at(iDial + v_offsets.at(detectorIndex)) ){
                LogFatal << "GOT WEIGHT FROM CACHE: " << dialCachePtr->cachedWeight << std::endl;
            }
            else{
                if(not currentDial->GetApplyOnlyOnMapPtr()->empty()){
                    LogFatal << "Apply only on conditions:" << std::endl;
                    for(const auto& applyCondition : *currentDial->GetApplyOnlyOnMapPtr()){
                        if(not GenericToolbox::doesElementIsInVector(
                            event->GetEventVarInt(applyCondition.first),
                            applyCondition.second)
                            ){
                            LogError << " -> NOT OK:" << applyCondition.first << ": didn't found \"" << event->GetEventVarInt(applyCondition.first) << "\" in " << GenericToolbox::parseVectorAsString(applyCondition.second) << std::endl;
                        }
                        else{
                            LogWarning << " -> OK: " << applyCondition.first << ": did found \"" << event->GetEventVarInt(applyCondition.first) << "\" in " << GenericToolbox::parseVectorAsString(applyCondition.second) << std::endl;
                        }
                    }
                }
                if(not currentDial->GetDontApplyOnMapPtr()->empty()){
                    LogFatal << "Don't apply on conditions:" << std::endl;
                    for(const auto& applyCondition : *currentDial->GetDontApplyOnMapPtr()){
                        if(GenericToolbox::doesElementIsInVector(
                            event->GetEventVarInt(applyCondition.first),
                            applyCondition.second)
                            ){
                            LogWarning << " -> OK: " << applyCondition.first << ": didn't found \"" << event->GetEventVarInt(applyCondition.first) << "\" in " << GenericToolbox::parseVectorAsString(applyCondition.second) << std::endl;
                        }
                        else{
                            LogError << " -> NOT OK:" << applyCondition.first << ": did found \"" << event->GetEventVarInt(applyCondition.first) << "\" in " << GenericToolbox::parseVectorAsString(applyCondition.second) << std::endl;
                        }
                    }
                }
            }
            throw std::runtime_error(GET_VAR_NAME_VALUE(currentDialWeight));
        }

        // Adding weight
        totalEventWeight *= currentDialWeight;

    }

    if(m_do_cap_weights){
        totalEventWeight = totalEventWeight > m_weight_cap ? m_weight_cap : totalEventWeight;
    }

    event -> AddEvWght(totalEventWeight);
}

void XsecParameters::AddDetector(const std::string& detectorName_, const std::string& configFilePath_)
{
    LogInfo << "Adding detector " << detectorName_ << " for " << m_name << std::endl;
    std::fstream f;
    LogInfo << "Opening config file " << configFilePath_ << std::endl;
    f.open(configFilePath_, std::ios::in);

    json j;
    f >> j;

    std::string input_dir = std::string(std::getenv("XSLLHFITTER"))
                            + j["input_dir"].get<std::string>();
    LogInfo << "Adding the following dials." << std::endl;

    std::vector<int> global_dimensions;
    if(j.find("subject_id") != j.end()){
        std::vector<int> global_dimensions = j["dimensions"].get<std::vector<int>>();
    }

    std::vector<XsecDial> xsecDialList;
    for(const auto& dialConfig : j["dials"]){
        if(dialConfig["use"] == true){
            LogInfo << "Setting up " << dialConfig["name"] << " dial..." << std::endl;
            XsecDial xsecDial(dialConfig["name"]);

            if(dialConfig.find("splines") != dialConfig.end()){
                std::string fname_splines = input_dir + dialConfig["splines"].get<std::string>();
                std::string fname_binning = input_dir + dialConfig["binning"].get<std::string>();;
                xsecDial.SetBinning(fname_binning);
                xsecDial.ReadSplines(fname_splines);
                if(xsecDial.GetSplineList().empty() and xsecDial.GetSplinePtrList().empty()){
                    LogAlert << "No splines has been found in " << fname_splines << ". Skipping..." << std::endl;
                    continue;
                }
            }
            else if(dialConfig.find("is_normalization_dial") != dialConfig.end() and bool(dialConfig["is_normalization_dial"])){
                xsecDial.SetIsNormalizationDial(true);
            }
            else{
                LogError << "Don't have splines and is not tagged as a normalization systematic. Skipping..." << std::endl;
                continue;
            }

            xsecDial.SetNominal(dialConfig["nominal"]);
            xsecDial.SetStep(dialConfig["step"]);
            xsecDial.SetLimitLo(dialConfig["limit_lo"]);
            xsecDial.SetLimitHi(dialConfig["limit_hi"]);
            if(dialConfig.find("prior") == dialConfig.end()){
                LogWarning << "Prior value parameter for dial " << dialConfig["name"]
                           << " not found. Nominal will be used as prior." << std::endl
                           << "Please consider setting this parameter in the future.";
                xsecDial.SetPrior(xsecDial.GetNominal());
            }
            else{
//                xsecDial.SetPrior(dialConfig["prior"]);
                xsecDial.SetPrior(xsecDial.GetNominal());
            }

            if(dialConfig.find("apply_only_on") != dialConfig.end()){
                std::map<std::string, std::vector<int>> applyOnlyOnMap;
                auto conditions = dialConfig["apply_only_on"].items();
                for (auto condition = conditions.begin(); condition != conditions.end(); ++condition){
                    applyOnlyOnMap[condition.key()] = dialConfig["apply_only_on"][condition.key()].get<std::vector<int>>();
                }
                xsecDial.SetApplyOnlyOnMap(applyOnlyOnMap);
            }
            if(dialConfig.find("dont_apply_on") != dialConfig.end()){
                std::map<std::string, std::vector<int>> dontApplyOnMap;
                auto conditions = dialConfig["dont_apply_on"].items();
                for (auto condition = conditions.begin(); condition != conditions.end(); ++condition){
                    dontApplyOnMap[condition.key()] = dialConfig["dont_apply_on"][condition.key()].get<std::vector<int>>();
                }
                xsecDial.SetDontApplyOnMap(dontApplyOnMap);
            }
            if(dialConfig.find("apply_condition") != dialConfig.end()){
                std::string applyCondition = dialConfig["apply_condition"].get<std::string>();
                xsecDial.SetApplyCondition(applyCondition);
            }

            xsecDial.Print(false);

            if(not global_dimensions.empty()){
                std::vector<int> dimensions = dialConfig.value("dimensions", global_dimensions);
                xsecDial.SetDimensions(dimensions);
            }

            xsecDialList.emplace_back(xsecDial);
        }
        else{
            LogAlert << "Skipping " << dialConfig["name"] << " dial (\"use\" == false)..." << std::endl;
        }
    }

    v_detectors.emplace_back(detectorName_);
    _dialsList_.emplace_back(xsecDialList);
}

void XsecParameters::SetEnableZeroWeightFenceGate(bool enableZeroWeightFenceGate_){
    _enableZeroWeightFenceGate_ = enableZeroWeightFenceGate_;
}

int XsecParameters::GetDetectorIndex(const std::string& detectorName_){
    int detectorIndex = GenericToolbox::findElementIndex(detectorName_, v_detectors);
    if(detectorIndex == -1){
        LogFatal << "Invalid detector name: " << detectorName_ << std::endl;
        throw std::logic_error("Invalid detector name.");
    }
    return detectorIndex;
}

std::vector<XsecDial> XsecParameters::GetDetectorDials(const std::string& detectorName_){
    return _dialsList_.at(GetDetectorIndex(detectorName_));
}
