//
// Created by Adrien Blanchet on 17/05/2023.
//

#include "ConfigUtils.h"
#include "GundamGlobals.h"
#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.Map.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include <TMatrixDEigen.h>

#include <string>
#include <vector>
#include <map>


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[" << FILENAME << "]"; });
#endif


using namespace GenericToolbox::ColorCodes;

CmdLineParser clParser;


// template function that handles every case when the requested TObject isn't found.
void readMatrix( const std::string& title_, TMatrixD* matrix_ );



int main(int argc, char** argv){

  // --------------------------
  // Greetings:
  // --------------------------
  GundamGreetings g;
  g.setAppName("fit output reader tool");
  g.hello();

  // --------------------------
  // Read Command Line Args:
  // --------------------------

  clParser.getDescription() << " > " << FILENAME << " is a program that reads in the output files of gundamFitter." << std::endl;

  LogInfo << clParser.getDescription().str() << std::endl;

  clParser.addDummyOption("Options");
  clParser.addOption("fitFiles", {"-f", "--file"}, "Specify path to fitter output files", -1);
  clParser.addOption("verbose", {"-v", "--verbose"}, "Set the verbosity level", 1, true);
  clParser.addOption("showCorrelationsWith", {"--show-correlations-with"}, "Show all correlation coefficients of a given par wrt others", -1);
  clParser.addOption("outFolder", {"-o", "--out-folder"}, "Set output folder where files will be writen (when extractDataToDisk option is triggered)", 1);

  clParser.addDummyOption("Triggers");
  clParser.addTriggerOption("extractDataToDisk", {"-e", "--extract"}, "Export data to output files");
  clParser.addTriggerOption("showParList", {"--show-par-list"}, "Show parameters list");

  clParser.addDummyOption();

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;


  LogThrowIf( clParser.getNbValueSet("fitFiles") == 0, "no fit files provided" );


  for( auto& file : clParser.getOptionValList<std::string>("fitFiles") ){
    LogInfo << std::endl << magentaLightText << "Opening: " << resetColor << file << std::endl;
    LogScopeIndent;

    auto outDir{GenericToolbox::joinPath(
        (clParser.isOptionTriggered("outFolder") ?
          clParser.getOptionVal<std::string>("outFolder"):
          GenericToolbox::getFolderPath(file).empty() ? "./" : GenericToolbox::getFolderPath(file)
        ),
        GenericToolbox::getFileName(file, false)
    )};

    if( clParser.isOptionTriggered("extractDataToDisk") ){
      LogWarning << "Output files will be writen under: " << outDir << std::endl;
    }

    auto f = std::make_shared<TFile>(file.c_str());
    LogContinueIf(not GenericToolbox::doesTFileIsValid(f.get()), "Could not open \"" << file << "\"");

    std::string gundamDirName{};
    if     ( f->Get("gundam") != nullptr ){ gundamDirName = "gundam"; }
    else if( f->Get("gundamFitter") != nullptr ){ gundamDirName = "gundamFitter"; } // legacy
    else if( f->Get("gundamCalcXsec") != nullptr ){ gundamDirName = "gundamCalcXsec"; } // legacy
    LogContinueIf(gundamDirName.empty(), "Not a gundam fitter output file.");


    {
      LogInfo << "Fetching runtime info..." << std::endl; LogScopeIndent;
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(),
          {{GenericToolbox::joinPath(gundamDirName, "runtime/commandLine_TNamed")},
           {GenericToolbox::joinPath(gundamDirName, "commandLine_TNamed")}},
          []( TNamed *obj_ ){
            LogInfo << blueLightText << "Command line: " << resetColor << obj_->GetTitle() << std::endl;
          });
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/date_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "Date: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/user_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "User: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/pwd_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "Directory: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/host_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "Hostname: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/os_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "OS: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/dist_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "Distribution: " << resetColor << obj_->GetTitle() << std::endl; }
      );
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "runtime/arch_TNamed"),
          []( TNamed *obj_ ){ LogInfo << blueLightText << "Architecture: " << resetColor << obj_->GetTitle() << std::endl; }
      );
    }

    {
      LogInfo << "Fetching build info..." << std::endl; LogScopeIndent;
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(),
          {{GenericToolbox::joinPath(gundamDirName, "build/version_TNamed")},
           {GenericToolbox::joinPath(gundamDirName, "version_TNamed")}},
          []( TNamed *obj_ ){
            LogInfo << blueLightText << "Generated with GUNDAM version: " << resetColor << obj_->GetTitle()
                    << std::endl;
          });
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "build/root/version_TNamed"),
          []( TNamed *obj_ ){
            LogScopeIndent;
            LogInfo << blueLightText << "GUNDAM built against ROOT version: " << resetColor << obj_->GetTitle() << std::endl;
          });
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "build/root/date_TNamed"),
          []( TNamed *obj_ ){
            LogScopeIndent;
            LogInfo << blueLightText << "ROOT release date: " << resetColor << obj_->GetTitle() << std::endl;
          });
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(), GenericToolbox::joinPath(gundamDirName, "build/root/install_TNamed"),
          []( TNamed *obj_ ){
            LogScopeIndent;
            LogInfo << blueLightText << "ROOT install path: " << resetColor << obj_->GetTitle() << std::endl;
          });
    }


    if( clParser.isOptionTriggered("extractDataToDisk") ){
      LogInfo << "Extract data to disk..." << std::endl; LogScopeIndent;
      GundamUtils::ObjectReader::readObject<TNamed>(
          f.get(),
          {{GenericToolbox::joinPath(gundamDirName, "config/unfoldedJson_TNamed")},
           {GenericToolbox::joinPath(gundamDirName, "config_TNamed")},
           {GenericToolbox::joinPath(gundamDirName, "unfoldedConfig_TNamed")}},
          [&](TNamed* obj_){
            if( not GenericToolbox::isDir(outDir) ){ GenericToolbox::mkdir(outDir); }
            auto outConfigPath = GenericToolbox::joinPath(outDir, "config.json");
            LogInfo << blueLightText << "Writing unfolded config under: " << resetColor << outConfigPath << std::endl;
            GenericToolbox::dumpStringInFile( outConfigPath, obj_->GetTitle() );
          });
    }



    // FitterEngine/propagator
    do {
      auto pathPropagator{GenericToolbox::joinPath("FitterEngine", "propagator")};
      if( not GundamUtils::ObjectReader::readObject(f.get(), pathPropagator) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPropagator << resetColor << std::endl;
      LogScopeIndent;

      GundamUtils::ObjectReader::readObject<TMatrixT<double>>(f.get(), GenericToolbox::joinPath(pathPropagator, "globalCovarianceMatrix_TMatrixD"), [&](TMatrixT<double>* matrix){
        readMatrix("Global prior covariance matrix", matrix);
      });

      for( auto& parSet : GenericToolbox::lsSubDirTDirectory( f->Get<TDirectory>(pathPropagator.c_str()) ) ){

        if( f->Get(GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixD").c_str()) ){
          GundamUtils::ObjectReader::readObject<TMatrixT<double>>(
              f.get(),
              GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixD"),
              [&](TMatrixT<double>* matrix){
                readMatrix(parSet + " prior covariance matrix", matrix);
              });
        }
        else if( f->Get(GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixDSym").c_str()) ){
          GundamUtils::ObjectReader::readObject<TMatrixTSym<double>>(
              f.get(),
              GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixDSym"),
              [&](TMatrixTSym<double>* matrix){
                readMatrix(parSet + " prior covariance matrix", (TMatrixD*) matrix);
              });
        }
        else{
          LogError << "Could not find covariance matrix for " << parSet << std::endl;
        }
      }

    } while( false ); // allows to skip if not found

    /// FitterEngine/preFit
    do {
      auto pathPreFit{GenericToolbox::joinPath("FitterEngine", "preFit")};
      if( not GundamUtils::ObjectReader::readObject(f.get(), pathPreFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPreFit << resetColor << std::endl;
      LogScopeIndent;

      GundamUtils::ObjectReader::readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "llhState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.isOptionTriggered("verbose") ){
          LogInfo << blueLightText << "Pre-fit Likelihood state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( clParser.isOptionTriggered("extractDataToDisk") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
          if( not GenericToolbox::isDir( outSubDir ) ){ GenericToolbox::mkdir( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
          LogInfo << blueLightText << "Writing pre-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });

      GundamUtils::ObjectReader::readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "parState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.getOptionVal("verbose", 0) >= 1 ){
          LogInfo << blueLightText << "Pre-fit parameters state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( clParser.isOptionTriggered("extractDataToDisk") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
          if( not GenericToolbox::isDir( outSubDir ) ){ GenericToolbox::mkdir( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".json");
          LogInfo << blueLightText << "Writing pre-fit LLH parameter injector under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });

    } while( false ); // allows to skip if not found

    /// FitterEngine/postFit
    do {
      auto pathPostFit{GenericToolbox::joinPath("FitterEngine", "postFit")};
      if( not GundamUtils::ObjectReader::readObject(f.get(), pathPostFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPostFit << resetColor << std::endl;
      LogScopeIndent;

      std::string minimizationAlgo{};
      GundamUtils::ObjectReader::readObject<TTree>(f.get(), GenericToolbox::joinPath(pathPostFit, "bestFitStats"), [&](TTree* tree){
        tree->GetEntry(0);

        bool converged{tree->GetLeaf("fitConverged")->GetValue() == 1};
        LogInfo << (converged? greenLightText: redLightText) << "Did the fit converge? " << (converged ? "yes": "no") << resetColor << std::endl;

        int statusCode{int(tree->GetLeaf("fitStatusCode")->GetValue())};
        LogInfo << blueLightText << "Fit status code: " << resetColor;
        LogInfo << ( GenericToolbox::isIn( statusCode, GundamUtils::minuitStatusCodeStr ) ? GundamUtils::minuitStatusCodeStr.at(statusCode) : std::to_string(statusCode) );
        LogInfo << std::endl;
      });
      GundamUtils::ObjectReader::readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "llhState_TNamed"), [&](TNamed* injectorStr){
        LogInfo << blueLightText << "Post-fit Likelihood state: " << resetColor << injectorStr->GetTitle() << std::endl;
        if( clParser.isOptionTriggered("extractDataToDisk") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
          if( not GenericToolbox::isDir( outSubDir ) ){ GenericToolbox::mkdir( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
          LogInfo << blueLightText << "Writing post-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });
      GundamUtils::ObjectReader::readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "parState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.getOptionVal("verbose", 0) >= 1 ){
          LogInfo << blueLightText << "Post-fit parameters state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( clParser.isOptionTriggered("extractDataToDisk") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
          if( not GenericToolbox::isDir( outSubDir ) ){ GenericToolbox::mkdir( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".json");
          LogInfo << blueLightText << "Writing post-fit LLH parameter injector under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });

    } while( false ); // allows to skip if not found

    /// Multiple entries
    if( clParser.isOptionTriggered("showParList") ){
      LogInfo << cyanLightText << "Listing defined parameters:" << resetColor << std::endl;

      std::string pathPropagator{"FitterEngine/propagator"};
      for( auto& parSetDir : GenericToolbox::lsSubDirTDirectory( f->Get<TDirectory>(pathPropagator.c_str()) ) ){
        auto parSetPath = GenericToolbox::joinPath( pathPropagator, parSetDir, "parameters" );
        for( auto& parEntry : GenericToolbox::lsSubDirTDirectory( f->Get<TDirectory>(parSetPath.c_str()) ) ){
          LogScopeIndent;
          GundamUtils::ObjectReader::readObject<TNamed>( f.get(), GenericToolbox::joinPath( parSetPath, parEntry, "fullTitle_TNamed" ), [&](TNamed* obj){
            bool isEnabled{false};
            GundamUtils::ObjectReader::readObject<TNamed>( f.get(), GenericToolbox::joinPath( parSetPath, parEntry, "isEnabled_TNamed" ), [&](TNamed* obj){
              isEnabled = GenericToolbox::toBool(obj->GetTitle());
            });
            if( isEnabled ){
              LogInfo << obj->GetTitle() << std::endl;
            }
          });
        }
      }

    }

    if( clParser.isOptionTriggered("showCorrelationsWith") ){
      LogInfo << "Looking for strongest correlations with " << clParser.getOptionVal<std::string>("showCorrelationsWith", 0) << std::endl;
      auto pathPostFit{"FitterEngine/postFit"};

      GundamUtils::ObjectReader::readObject<TDirectory>( f.get(), pathPostFit, [&](TDirectory* postFitDir_){
        GundamUtils::ObjectReader::readObject<TDirectory>( postFitDir_, {{"Hesse"}, {"Migrad"}}, [&](TDirectory* hesseDir_){
          GundamUtils::ObjectReader::readObject<TH2D>( hesseDir_, "hessian/postfitCorrelationOriginal_TH2D", [&](TH2D* cor_){

            // clParser.getOptionVal<std::string>("showCorrelationsWith", 0)
            int selectedParIndex{-1};
            for( int iPar = 1 ; iPar <= cor_->GetNbinsX() ; iPar++ ){
              if( cor_->GetXaxis()->GetBinLabel(iPar) == clParser.getOptionVal<std::string>("showCorrelationsWith", 0) ){
                selectedParIndex = iPar;
                break;
              }
            }

            if( selectedParIndex == -1 ){
              LogError << "Could not find selected parameter" << std::endl;
              return;
            }

            struct CorrelationEntry{
              std::string parTitle{};
              double correlation{};

              CorrelationEntry() = default;
              CorrelationEntry(std::string parTitle_, double correlation_) : parTitle(std::move(parTitle_)), correlation(correlation_) {}
            };
            std::vector<CorrelationEntry> corrDict{};
            corrDict.reserve(cor_->GetNbinsX()-1);
            for( int iPar = 1 ; iPar <= cor_->GetNbinsX() ; iPar++ ){
              corrDict.emplace_back(cor_->GetXaxis()->GetBinLabel(iPar), cor_->GetBinContent(selectedParIndex, iPar));
            }

            GenericToolbox::sortVector(corrDict, [](const CorrelationEntry& a, const CorrelationEntry& b){
              return std::abs( a.correlation ) > std::abs( b.correlation );
            });

            GenericToolbox::TablePrinter t;
            t << "Fit parameter" << GenericToolbox::TablePrinter::NextColumn;
            t << "Correlation" << GenericToolbox::TablePrinter::NextLine;

            for( auto& corrEntry : corrDict ){
              t << corrEntry.parTitle << GenericToolbox::TablePrinter::NextColumn;
              t << corrEntry.correlation * 100 << " %" << GenericToolbox::TablePrinter::NextLine;
            }

            t.printTable();

          });
        });
      });

    }

    LogInfo << "Closing " << f->GetPath() << std::endl;
    f->Close();
  }

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  return EXIT_SUCCESS;
}


void readMatrix( const std::string& title_, TMatrixD* matrix_ ){
  LogInfo << blueLightText << title_ << resetColor << " dimensions are " << matrix_->GetNrows() << "x" << matrix_->GetNcols() << std::endl;

  LogScopeIndent;
  LogInfo << GET_VAR_NAME_VALUE(matrix_->Determinant()) << std::endl;
  LogInfo << GET_VAR_NAME_VALUE(matrix_->IsSymmetric()) << std::endl;

  if( clParser.isOptionTriggered("verbose") ){
    LogWarning << "Decomposing matrix..." << std::endl;
    TMatrixDEigen decompMatrix(*matrix_);

    double minEigen{std::nan("")};
    double maxEigen{std::nan("")};
    for( int i=0 ; i < decompMatrix.GetEigenValues().GetNcols() ; i++ ){
      minEigen = std::min(decompMatrix.GetEigenValues()[i][i], minEigen);
      maxEigen = std::max(decompMatrix.GetEigenValues()[i][i], maxEigen);
    }

    LogInfo << "Conditioning: " << minEigen << "/" << maxEigen
            << " = " << minEigen/maxEigen << std::endl;
  }
}
