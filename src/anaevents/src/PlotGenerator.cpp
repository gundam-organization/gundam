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
#include "PlotGenerator.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[PlotGenerator]");
})


PlotGenerator::PlotGenerator() { this->reset(); }
PlotGenerator::~PlotGenerator() { this->reset(); }

void PlotGenerator::reset() {

  defaultColorWheel = {
    kGreen-3, kTeal+3, kAzure+7,
    kCyan-2, kBlue-7, kBlue+2,
    kOrange+1, kOrange+9, kRed+2, kPink+9
  };

}

void PlotGenerator::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void PlotGenerator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ not set." << std::endl;
    throw std::logic_error("_config_ not set.");
  }

  _varDictionary_ = JsonUtils::fetchValue(_config_, "varDictionnaries", nlohmann::json());
  _canvasParameters_ = JsonUtils::fetchValue(_config_, "canvasParameters", nlohmann::json());
  _histogramsDefinition_ = JsonUtils::fetchValue(_config_, "histogramsDefinition", nlohmann::json());

}

std::map<std::string, TH1D *> PlotGenerator::getBufferHistogramList() const {
  return _bufferHistogramList_;
}
std::map<std::string, TCanvas *> PlotGenerator::getBufferCanvasList() const {
  return _bufferCanvasList_;
}
std::map<std::string, std::map<std::string, std::vector<TH1D *>>> PlotGenerator::getHistsToStack() const {
  return _histsToStack_;
}
std::map<std::string, std::map<std::string, std::vector<TH1D *>>> PlotGenerator::getCompHistsToStack() const {
  return _compHistsToStack_;
}

void PlotGenerator::generateSamplePlots(const std::vector<AnaSample> &sampleList_, TDirectory *saveTDirectory_) {

  LogWarning << "Generating sample plots..." << std::endl;

  this->generateSampleHistograms(sampleList_, GenericToolbox::mkdirTFile(saveTDirectory_, "histograms"));
  this->generateCanvas(_histsToStack_, GenericToolbox::mkdirTFile(saveTDirectory_, "canvas"));

}
void PlotGenerator::generateSampleHistograms(const std::vector<AnaSample> &sampleList_, TDirectory *saveDir_){

  if( _histogramsDefinition_.empty() ){
    LogError << "No histogram has been defined." << std::endl;
    return;
  }

  auto* lastDir = gDirectory;
  if(saveDir_ != nullptr ){
    saveDir_->cd();
    LogInfo << "Samples plots will be writen in: " << saveDir_->GetPath() << std::endl;
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

              _bufferHistogramList_[histPath]->SetLineWidth(2);
              _bufferHistogramList_[histPath]->SetMarkerStyle(kFullDotLarge);

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
            if(saveDir_ != nullptr){
              GenericToolbox::mkdirTFile(saveDir_, histDirPath )->cd();
              _bufferHistogramList_[histPath]->Write( histType.c_str() );
              saveDir_->cd();
            }
            _bufferHistogramList_[histPath]->SetName( histDirPath.c_str() );

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
void PlotGenerator::generateCanvas(const std::map<std::string, std::map<std::string, std::vector<TH1D*>>>& histsToStack_, TDirectory *saveDir_, bool stackHist_){

  auto* lastDir = gDirectory;
  if(saveDir_ != nullptr ){
    saveDir_->cd();
    LogInfo << "Samples plots will be writen in: " << saveDir_->GetPath() << std::endl;
  }

  int canvasHeight = JsonUtils::fetchValue(_canvasParameters_, "height", 700);
  int canvasWidth = JsonUtils::fetchValue(_canvasParameters_, "width", 1200);
  int canvasNbXplots = JsonUtils::fetchValue(_canvasParameters_, "nbXplots", 3);
  int canvasNbYplots = JsonUtils::fetchValue(_canvasParameters_, "nbYplots", 2);
  _bufferCanvasList_.clear();

  // Canvas builder
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

      bool isPlotInitialized = false;

      // process mc part
      std::vector<TH1D*> mcSampleHistAccumulatorList;
      if( not mcSampleHistList.empty() ){

        if( stackHist_ ){
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
            if( not isPlotInitialized ){
              mcSampleHistAccumulatorList[iHist]->GetYaxis()->SetRangeUser( minYValue,mcSampleHistAccumulatorList[iHist]->GetMaximum()*1.2 );
              mcSampleHistAccumulatorList[iHist]->Draw("HIST");
              isPlotInitialized = true;
            }
            else {
              mcSampleHistAccumulatorList[iHist]->Draw("HISTSAME");
            }
          }
        }
        else{
          // Just draw each hist on the same plot
          for( auto & mcHist : mcSampleHistList ){
            if( not isPlotInitialized ){
              if( mcSampleHistList.size() == 1 ){ mcHist->Draw("EP"); }
              else{
                mcHist->Draw("HIST P");
              } // don't draw multiple error bars
              isPlotInitialized = true;
            }
            else{
              if( mcSampleHistList.size() == 1 ){ mcHist->Draw("EPSAME"); }
              else{ mcHist->Draw("HIST P SAME"); }
            }
          } // mcHist
        } // stack?


      } // mcHistList empty?

      // Draw the data hist on top
      if( dataSampleHist != nullptr ){
        std::string originalTitle = dataSampleHist->GetTitle(); // title can be used for figuring out the type of the histogram
        dataSampleHist->SetTitle("Data");
        if( isPlotInitialized ){
          dataSampleHist->Draw("EPSAME");
        }
        else{
          dataSampleHist->Draw("EP");
          isPlotInitialized = true;
        }
        dataSampleHist->SetTitle(originalTitle.c_str()); // restore
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

  }

  // Write
  for( auto& canvas : _bufferCanvasList_ ){
    auto pathSplit = GenericToolbox::splitString(canvas.first, "/");
    std::string folderPath = GenericToolbox::joinVectorString(pathSplit, "/", 0, -1);
    std::string canvasName = pathSplit.back();
    if(saveDir_ != nullptr){
      GenericToolbox::mkdirTFile(saveDir_, folderPath )->cd();
      canvas.second->Write( canvasName.c_str() );
      canvas.second->SetName( ( saveDir_->GetPath() + canvas.first ).c_str() ); // full path to avoid ROOT deleting
      saveDir_->cd();
    }
  }

  lastDir->cd();

}

void PlotGenerator::generateComparisonPlots(
  const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackOther_,
  const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackReference_,
  TDirectory *saveDir_){

  LogWarning << "Generating comparison plots..." << std::endl;

  this->generateComparisonHistograms(histsToStackOther_, histsToStackReference_, GenericToolbox::mkdirTFile(saveDir_, "histograms"));
  this->generateCanvas(_compHistsToStack_, GenericToolbox::mkdirTFile(saveDir_, "canvas"), false);

}
void PlotGenerator::generateComparisonHistograms(
  const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackOther_,
  const std::map<std::string, std::map<std::string, std::vector<TH1D *>>> &histsToStackReference_,
  TDirectory *saveDir_) {

  auto* curDir = gDirectory;

  if(saveDir_ != nullptr){
    saveDir_->cd();
    LogInfo << "Comparison histograms will be writen in: " << saveDir_->GetPath() << std::endl;
  }
  _compHistsToStack_.clear();

  for( const auto& stackEntry : histsToStackOther_ ){

    if( not GenericToolbox::doesKeyIsInMap(stackEntry.first, histsToStackReference_) ){
      continue; // skip if no reference
    }
    const auto& refSampleCollections = histsToStackReference_.at(stackEntry.first);

    _compHistsToStack_[stackEntry.first] = std::map<std::string, std::vector<TH1D*>>();
    for( const auto& sampleCollection : stackEntry.second ){

      if( not GenericToolbox::doesKeyIsInMap(sampleCollection.first, refSampleCollections) ){
        continue; // skip if no reference
      }
      const auto& refSampleCollection = refSampleCollections.at(sampleCollection.first);

      _compHistsToStack_[stackEntry.first][sampleCollection.first] = std::vector<TH1D*>();
      auto& compHistList = _compHistsToStack_[stackEntry.first][sampleCollection.first];

      for( const auto& otherHist : sampleCollection.second){

        if( GenericToolbox::doesStringEndsWithSubstring(otherHist->GetTitle(), "data") ){
          // data hist are supposed to be static (not affected by any parameter)
          continue;
        }

        TH1D* refHist{nullptr};
        for( const auto& refHistCandidate : refSampleCollection){
          if(     std::string(otherHist->GetName()) == std::string(refHistCandidate->GetName()) // same path?
                  and otherHist->GetNbinsX() == refHistCandidate->GetNbinsX() ){
            refHist = refHistCandidate;
            break;
          }
        }
        if( refHist == nullptr ){
          // if corresponding ref hist not found, skip
          break;
        }

        compHistList.emplace_back( (TH1D*) otherHist->Clone() );
        for( int iBin = 0 ; iBin <= compHistList.back()->GetNbinsX()+1 ; iBin++ ){

          if( refHist->GetBinContent(iBin) == 0 ){
            // no division by 0
            compHistList.back()->SetBinContent( iBin, 0 );
          }

          double binContent = compHistList.back()->GetBinContent( iBin );
          binContent /= refHist->GetBinContent(iBin);
          binContent -= 1;
          binContent *= 100.;
          compHistList.back()->SetBinContent( iBin, binContent );
          compHistList.back()->SetBinError( iBin, otherHist->GetBinError(iBin) / otherHist->GetBinContent(iBin) * 100 );

        } // iBin
        compHistList.back()->GetYaxis()->SetTitle("Relative Deviation (%)");

        // Y axis
        double Ymin = 0;
        double Ymax = 0;
        for(int iBin = 0 ; iBin <= compHistList.back()->GetNbinsX() + 1 ; iBin++){
          double val = compHistList.back()->GetBinContent(iBin);
          double error = compHistList.back()->GetBinError(iBin);
          if(val + error > Ymax){
            Ymax = val + error;
          }
          if(val - error < Ymin){
            Ymin = val - error;
          }
        }

        // Add 20% margin
        Ymin += Ymin*0.2;
        Ymax += Ymax*0.2;

        // Force showing Y=0
        if(Ymin > -0.2) Ymin = -0.2;
        if(Ymax < 0.2) Ymax = 0.2;

        compHistList.back()->GetYaxis()->SetRangeUser(Ymin, Ymax);

        if( saveDir_ != nullptr ){
          auto splitPath = GenericToolbox::splitString( compHistList.back()->GetName(), "/" );
          std::string histogramFolder = GenericToolbox::joinVectorString(splitPath, "/", 0, -1);
          std::string histogramName = splitPath.back();
          GenericToolbox::mkdirTFile( saveDir_, histogramFolder )->cd();
          compHistList.back()->Write( histogramName.c_str() );
          saveDir_->cd();
        }
      } // iHist

    } // sample

  } // entry

  curDir->cd();

}

