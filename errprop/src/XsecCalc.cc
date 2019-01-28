#include "XsecCalc.hh"

XsecCalc::XsecCalc(const std::string& json_config)
    : num_toys(0)
{
    selected_events = new FitObj(json_config, "selectedEvents", false);
    true_events = new FitObj(json_config, "trueEvents", true);
}

XsecCalc::~XsecCalc()
{
    delete selected_events;
    delete true_events;
}

void XsecCalc::ReweightNominal()
{
    selected_events -> ReweightNominal();
    true_events -> ReweightNominal();
}

void XsecCalc::GenerateToys()
{
}

void XsecCalc::GenerateToys(const int ntoys)
{
}
