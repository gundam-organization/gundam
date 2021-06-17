//
// Created by Adrien BLANCHET on 16/06/2021.
//

#include "string"
#include "vector"
#include "sstream"

#include "TCanvas.h"
#include "TH1D.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

#include "JsonUtils.h"
#include "SamplePlotGenerator.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[SamplePlotGenerator]");
})


SamplePlotGenerator::SamplePlotGenerator() { this->reset(); }
SamplePlotGenerator::~SamplePlotGenerator() { this->reset(); }

void SamplePlotGenerator::reset() {

  defaultColorWheel = {
    kGreen-3, kTeal+3, kAzure+7,
    kCyan-2, kBlue-7, kBlue+2,
    kOrange+1, kOrange+9, kRed+2, kPink+9
  };

}

void SamplePlotGenerator::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void SamplePlotGenerator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ not set." << std::endl;
    throw std::logic_error("_config_ not set.");
  }

  _varDictionary_ = JsonUtils::fetchValue(_config_, "varDictionnaries", nlohmann::json());
  _canvasParameters_ = JsonUtils::fetchValue(_config_, "canvasParameters", nlohmann::json());
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());

}

std::map<std::string, TH1D *> SamplePlotGenerator::getBufferHistogramList() const {
  return _bufferHistogramList_;
}
std::map<std::string, TCanvas *> SamplePlotGenerator::getBufferCanvasList() const {
  return _bufferCanvasList_;
}

void SamplePlotGenerator::generateSamplePlots(const std::vector<AnaSample> &sampleList_, TDirectory *saveTDirectory_) {

  LogWarning << "Generating sample plots..." << std::endl;

  this->generateSampleHistograms(sampleList_, GenericToolbox::mkdirTFile(saveTDirectory_, "histograms"));
  this->generateCanvas(_histsToStack_, GenericToolbox::mkdirTFile(saveTDirectory_, "canvas"));

}
void SamplePlotGenerator::generateSampleHistograms(const std::vector<AnaSample> &sampleList_, TDirectory *saveTDirectory_){

  if( _histogramsDefinition_.empty() ){
    LogError << "No histogram has been defined." << std::endl;
    return;
  }

  auto* lastDir = gDirectory;
  if( saveTDirectory_ != nullptr ){
    saveTDirectory_->cd();
    LogInfo << "Samples plots will be writen in: " << saveTDirectory_->GetPath() << std::endl;
  }

  // Clearing the buffers: ROOT takes the ownership of the ptr anyway
  _bufferHistogramList_.clear();
  _histsToStack_.clear();

  int sampleCounter = -1;
  for( const auto& sample : sampleList_ ){
    sampleCounter++;

    short unsetSplitValueColor = kGray; // will increment if needed
    std::vector<std::string> varToPlotFoldersList; // List of plotted vars
    for( const auto& histDef : _histogramsDefinition_ ){

      // Parameters
      std::string varToPlot = JsonUtils::fetchValue<std::string>(histDef, "varToPlot");
      std::string varToPlotPrefix = JsonUtils::fetchValue(histDef, "prefix", "");
      std::vector<std::string> splitVars = JsonUtils::fetchValue(histDef, "splitVars", std::vector<std::string>{""});
      bool rescaleAsBinWidth = JsonUtils::fetchValue(histDef, "rescaleAsBinWidth", true);
      double rescaleBinFactor = JsonUtils::fetchValue(histDef, "rescaleBinFactor", 1.);

      // Check if the varToPlot has not already been plotted
      std::string folderCandidate = varToPlot + varToPlotPrefix;
      if( not GenericToolbox::doesElementIsInVector(folderCandidate, varToPlotFoldersList) ){
        varToPlotFoldersList.emplace_back(folderCandidate);
      }
      else{
        int index = 2;
        while( GenericToolbox::doesElementIsInVector(folderCandidate + "_" + std::to_string(index), varToPlotFoldersList) ){
          index++;
        }
        varToPlotFoldersList.emplace_back(folderCandidate + "_" + std::to_string(index));
      }

      std::stringstream ssVarToPlotPath;
      ssVarToPlotPath << sample.GetName() << "/";
      ssVarToPlotPath << varToPlotFoldersList.back() << "/";

      TH1D* lastDataHist{nullptr};
      for( const auto& splitVar : splitVars ){

        std::stringstream ssHistPath;
        ssHistPath << ssVarToPlotPath.str();

        std::stringstream ssCanvasPath;
        ssCanvasPath << varToPlotFoldersList.back() << "/";

        std::vector<int> splitVarValues;
        if( not splitVar.empty() ){
          ssHistPath << splitVar << "/";
          ssCanvasPath << splitVar << "/";
          for( const auto& event : sample.GetConstMcEvents() ){
            int splitValue = event.GetEventVarInt(splitVar);
            if( not GenericToolbox::doesElementIsInVector(splitValue, splitVarValues) ){
              splitVarValues.emplace_back(splitValue);
            }
          }
        }
        else{
          splitVarValues.emplace_back(0); // just for the loop
        }


        _histsToStack_[ssCanvasPath.str()][sample.GetName()] = std::vector<TH1D*>();

        // Loop over split vars
        int splitValueIndex = -1;
        for( const auto& splitValue : splitVarValues ){
          splitValueIndex++;

          std::string histDirPath = ssHistPath.str();
          if( not splitVar.empty() ){
            histDirPath += std::to_string(splitValue) + "/";
          }

          for( const std::string& histType: {"mc", "data"}){

            if( // don't build data histogram in those cases
              histType == "data" and
              ( not splitVar.empty()  // don't use split var for data
                or JsonUtils::fetchValue(histDef, "noData", false) // explicitly no data
              )
              ){
              continue;
            }

            std::string histPath = histDirPath + histType;
            _bufferHistogramList_[histPath] = nullptr;

            if( varToPlot == "Raw" ){
              if( histType == "mc" ) _bufferHistogramList_[histPath] = (TH1D*) sample.GetPredHisto().Clone();
              else if( histType == "data" ) _bufferHistogramList_[histPath] = (TH1D*) sample.GetDataHisto().Clone();
            }
            else {

              std::vector<double> xBinEdges;
              double xMin = JsonUtils::fetchValue(histDef, "xMin", std::nan("nan"));;
              double xMax = JsonUtils::fetchValue(histDef, "xMax", std::nan("nan"));


              if( JsonUtils::fetchValue(histDef, "useSampleBinning", false) ){

                int dimIndex = GenericToolbox::findElementIndex(varToPlot, sample.GetFitPhaseSpace());
                if( dimIndex == -1 ){
                  LogError << "Can't useSampleBinning since \"" << varToPlot
                           << "\" is not a dimension of the sample \"" << sample.GetName()
                           << "\": " << GenericToolbox::parseVectorAsString(sample.GetFitPhaseSpace()) << std::endl;
                  continue; // skip this sample
                }


                for( const auto& bin : sample.GetBinEdges() ){
                  auto edges = { bin.getBinLowEdge(dimIndex), bin.getBinHighEdge(dimIndex) };
                  for( const auto& edge : edges ){
                    if( ( xMin != xMin or xMin <= edge ) and ( xMax != xMax or xMax >= edge ) ){ // either NaN or in bounds
                      if( not GenericToolbox::doesElementIsInVector(edge, xBinEdges) ){
                        xBinEdges.emplace_back(edge);
                      }
                    }
                  }
                }
                if( xBinEdges.empty() ) continue; // skip
                std::sort(xBinEdges.begin(), xBinEdges.end()); // sort for ROOT

                _bufferHistogramList_[histPath] = new TH1D(
                  histPath.c_str(), histType.c_str(), int(xBinEdges.size()) - 1, &xBinEdges[0]
                );

                std::string xTitle = JsonUtils::fetchValue(histDef, "xTitle", varToPlot);
                _bufferHistogramList_[histPath]->GetXaxis()->SetTitle(xTitle.c_str() );

                std::string yTitle = "Counts";
                if( rescaleAsBinWidth ) yTitle += " (/bin width)";
                if( rescaleBinFactor != 1. ) yTitle += "*" + std::to_string(rescaleBinFactor);
                yTitle = JsonUtils::fetchValue(histDef, "yTitle", yTitle);
                _bufferHistogramList_[histPath]->GetYaxis()->SetTitle(yTitle.c_str() );

              }
              else{
                LogError << "Unsupported yet." << std::endl;
                throw std::logic_error("unsupported yet");
              }

              // Fill the histogram
              const std::vector<AnaEvent>* eventList{nullptr};
              if( histType == "mc" ){
                eventList = &sample.GetConstMcEvents();
              }
              else if( histType == "data" ){
                eventList = &sample.GetConstDataEvents();
              }
              if( eventList != nullptr ){
                for( const auto& event : *eventList ){
                  if( // Condition to fill:
                    splitVar.empty() // no split var: every event in the hist
                    or event.GetEventVarInt(splitVar) == splitValue // it's the corresponding splitValue
                    ){
                    _bufferHistogramList_[histPath]->Fill(event.GetEventVarAsDouble(varToPlot), event.GetEventWeight() );
                  }
                }
              }

              // Scaling
              if( histType == "mc" ){
                _bufferHistogramList_[histPath]->Scale(sample.GetNorm() );
              }
              else if( histType == "data" and sample.GetDataType() == DataType::kAsimov ){
                // Asimov is just MC, so it should be normalized the same way as MC
                _bufferHistogramList_[histPath]->Scale(sample.GetNorm() );
              }

              // Normalize by bin width
              for(int iBin = 0 ; iBin <= _bufferHistogramList_[histPath]->GetNbinsX() + 1 ; iBin++ ){
                double binContent = _bufferHistogramList_[histPath]->GetBinContent(iBin); // this is the real number of counts
                double errorFraction = TMath::Sqrt(binContent)/binContent; // error fraction will not change with renorm of bin width

                if( rescaleAsBinWidth ) binContent /= _bufferHistogramList_[histPath]->GetBinWidth(iBin);
                if( rescaleBinFactor != 1. ) binContent *= rescaleBinFactor;
                _bufferHistogramList_[histPath]->SetBinContent(iBin, binContent );
                _bufferHistogramList_[histPath]->SetBinError(iBin, binContent * errorFraction );
              }

            } // varToPlot is not Raw?

            if( histType == "data" ){
              _bufferHistogramList_[histPath]->SetLineColor(kBlack);
              _bufferHistogramList_[histPath]->SetMarkerColor(kBlack);
              _bufferHistogramList_[histPath]->SetMarkerStyle(kFullDotLarge);
              _bufferHistogramList_[histPath]->SetOption("EP");
              lastDataHist = _bufferHistogramList_[histPath];
            }
            else{


              if( splitVar.empty() ){
                _bufferHistogramList_[histPath]->SetLineColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
                _bufferHistogramList_[histPath]->SetFillColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
                _bufferHistogramList_[histPath]->SetMarkerColor(defaultColorWheel[sampleCounter % defaultColorWheel.size() ]);
              }
              else{

                // default
                std::string valueTitle = splitVar + " = " + std::to_string(splitValue);
                short valueColor = defaultColorWheel[splitValueIndex % defaultColorWheel.size() ];

                auto varDict = JsonUtils::fetchMatchingEntry(_varDictionary_, "name", splitVar); // does the cosmetic pars are configured?
                if( not varDict.empty() ){

                  if( varDict["dictionary"].is_null() ){
                    LogError << R"(Could not find "dictionary" key in JSON config for var: ")" << splitVar << "\"" << std::endl;
                    throw std::runtime_error("dictionary not found, by variable name found in JSON.");
                  }

                  // Look for the value we want
                  auto valDict = JsonUtils::fetchMatchingEntry(varDict["dictionary"], "value", splitValue);

                  valueTitle = JsonUtils::fetchValue(valDict, "title", valueTitle);
                  valueColor = JsonUtils::fetchValue(valDict, "color", unsetSplitValueColor);
                  if( valueColor == unsetSplitValueColor ) unsetSplitValueColor++; // increment for the next ones

                }

                _bufferHistogramList_[histPath]->SetTitle(valueTitle.c_str() );
                _bufferHistogramList_[histPath]->SetLineColor(valueColor );
                _bufferHistogramList_[histPath]->SetFillColor(valueColor );
                _bufferHistogramList_[histPath]->SetMarkerColor(valueColor );

              } // splitVar not empty

            } // is MC


            LogDebug << "Writing histogram: " << histPath << std::endl;
            if(saveTDirectory_ != nullptr) GenericToolbox::mkdirTFile(saveTDirectory_, "histograms/" + histDirPath)->cd();
            _bufferHistogramList_[histPath]->SetName(histType.c_str());
            _bufferHistogramList_[histPath]->Write(histType.c_str());
//            if(saveTDirectory_ != nullptr) saveTDirectory_->cd();

            _histsToStack_[ssCanvasPath.str()][sample.GetName()].emplace_back(_bufferHistogramList_[histPath]);

          } // mc or data
        } // splitVal

      } // splitVar

      // Adding data hist in var-split histogram list (for Canvas)
      if( lastDataHist != nullptr ){
        for( auto& histToStack : _histsToStack_ ){
          if( GenericToolbox::doesStringStartsWithSubstring( sample.GetName() + "/" + histToStack.first, ssVarToPlotPath.str())){
            if( not GenericToolbox::doesElementIsInVector(lastDataHist, histToStack.second[sample.GetName()]) ){
              histToStack.second[sample.GetName()].emplace_back((TH1D*) lastDataHist->Clone());
            }
          }
        } // _histsToStack_
      } // lastDataHist != nullptr


    } // histDef

  } // sample

  lastDir->cd();

}
void SamplePlotGenerator::generateCanvas(const std::map<std::string, std::map<std::string, std::vector<TH1D*>>>& histsToStack_, TDirectory *saveTDirectory_){

  auto* lastDir = gDirectory;
  if( saveTDirectory_ != nullptr ){
    saveTDirectory_->cd();
    LogInfo << "Samples plots will be writen in: " << saveTDirectory_->GetPath() << std::endl;
  }

  int canvasHeight = JsonUtils::fetchValue(_canvasParameters_, "height", 700);
  int canvasWidth = JsonUtils::fetchValue(_canvasParameters_, "width", 1200);
  int canvasNbXplots = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 3);
  int canvasNbYplots = JsonUtils::fetchValue(_canvasParameters_, "nbYplots", 2);
  _bufferCanvasList_.clear();

  for( const auto& hists : histsToStack_ ){

    int canvasIndex = 0;
    int iSampleSlot = 0;
    for( const auto& sampleHists : hists.second){
      iSampleSlot++;

      if(iSampleSlot > canvasNbXplots * canvasNbYplots ){
        canvasIndex++;
        iSampleSlot = 1;
      }

      std::string canvasName = "samples_n" + std::to_string(canvasIndex);
      std::string canvasPath = hists.first + canvasName;
      if( not GenericToolbox::doesKeyIsInMap(canvasPath, _bufferCanvasList_) ){
        _bufferCanvasList_[canvasPath] = new TCanvas( canvasPath.c_str(), canvasPath.c_str(), canvasWidth, canvasHeight );
        _bufferCanvasList_[canvasPath]->Divide(canvasNbXplots, canvasNbYplots);
      }
      _bufferCanvasList_[canvasPath]->cd(iSampleSlot);

      // separating histograms
      TH1D* dataSampleHist{nullptr};
      std::vector<TH1D*> mcSampleHistList;
      double minYValue = 1;
      for( auto& hist : sampleHists.second ){
        if( hist->GetTitle() == std::string("data") ){
          dataSampleHist = hist;
        }
        else{
          mcSampleHistList.emplace_back(hist);
          minYValue = std::min(minYValue, hist->GetMinimum(0));
        }
      }

      // process mc part
      std::vector<TH1D*> mcSampleHistAccumulatorList;
      if( not mcSampleHistList.empty() ){

        // Sorting histograms by norm (lowest stat first)
        std::function<bool(TH1D*, TH1D*)> aGoesFirst = [](TH1D* histA_, TH1D* histB_){
          return (  histA_->Integral(histA_->FindBin(0), histA_->FindBin(histA_->GetXaxis()->GetXmax()))
                    < histB_->Integral(histB_->FindBin(0), histB_->FindBin(histB_->GetXaxis()->GetXmax())) );
        };
        auto p = GenericToolbox::getSortPermutation( mcSampleHistList, aGoesFirst );
        mcSampleHistList = GenericToolbox::applyPermutation(mcSampleHistList, p);

        // Stacking histograms
        TH1D* histPileBuffer = nullptr;
        for( auto & hist : mcSampleHistList ){
          mcSampleHistAccumulatorList.emplace_back( (TH1D*) hist->Clone() ); // inheriting cosmetics as well
          if(histPileBuffer != nullptr){
            mcSampleHistAccumulatorList.back()->Add(histPileBuffer);
          }
          histPileBuffer = mcSampleHistAccumulatorList.back();
        }

        // Draw the stack
        int lastIndex = int(mcSampleHistAccumulatorList.size())-1;
        for( int iHist = lastIndex ; iHist >= 0 ; iHist-- ){
          if( iHist == lastIndex ){
            mcSampleHistAccumulatorList[iHist]->GetYaxis()->SetRangeUser( minYValue,mcSampleHistAccumulatorList[iHist]->GetMaximum()*1.2 );
            mcSampleHistAccumulatorList[iHist]->Draw("HIST");
          }
          else {
            mcSampleHistAccumulatorList[iHist]->Draw("HISTSAME");
          }
        }

      }

      // Draw the data hist on top
      if( dataSampleHist != nullptr ){
        dataSampleHist->SetTitle("Data");
        dataSampleHist->Draw("EPSAME");
      }

      // Legend
      double Xmax = 0.9;
      double Ymax = 0.9;
      double Xmin = 0.5;
      double Ymin = Ymax - 0.04*double(mcSampleHistList.size()+1);
      gPad->BuildLegend(Xmin,Ymin,Xmax,Ymax);

      if( not mcSampleHistAccumulatorList.empty() ) mcSampleHistAccumulatorList.back()->SetTitle(sampleHists.first.c_str()); // the actual title
      gPad->SetGridx();
      gPad->SetGridy();

      LogWarning << hists.first << ": " << sampleHists.first << " -> " << GenericToolbox::parseVectorAsString(sampleHists.second) << std::endl;
    } // sample
  } // canvas

  for( auto& canvas : _bufferCanvasList_ ){
    auto pathSplit = GenericToolbox::splitString(canvas.first, "/");
    std::string folderPath = GenericToolbox::joinVectorString(pathSplit, "/", 0, -1);
    std::string canvasName = pathSplit.back();
    if(saveTDirectory_ != nullptr) GenericToolbox::mkdirTFile( saveTDirectory_, "canvas/" + folderPath )->cd();
    canvas.second->Write( canvasName.c_str() );
  }

  lastDir->cd();

}



