#include "XsecCalc.hh"

XsecCalc::XsecCalc(const std::string& json_config)
{
    selected_events = new FitObj(json_config, "selectedEvents", false);
    true_events = new FitObj(json_config, "trueEvents", true);
}

XsecCalc::~XsecCalc()
{
    delete selected_events;
    delete true_events;
}
