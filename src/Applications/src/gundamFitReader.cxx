//
// Created by Adrien Blanchet on 17/05/2023.
//

#include "VersionConfig.h"
#include "ConfigUtils.h"
#include "GlobalVariables.h"
#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"

#include <TMatrixDEigen.h>

#include <string>
#include "vector"


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


using namespace GenericToolbox::ColorCodes;

CmdLineParser clParser;
bool quiet{false};


// template function that handles every case when the requested TObject isn't found.
template<typename T> bool readObject( TDirectory* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ = [](T*){} );
template<typename T> bool readObject( TDirectory* f_, const std::string& objPath_, const std::function<void(T*)>& action_ = [](T*){} ){ return readObject(f_, std::vector<std::string>{objPath_}, action_); }
bool readObject( TDirectory* f_, const std::string& objPath_){ return readObject<TObject>(f_, objPath_); }
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

  clParser.getDescription() << "gundamFitInfo is a program that reads in the output files of gundamFitter." << std::endl;

  LogInfo << clParser.getDescription().str() << std::endl;

  clParser.addDummyOption("Options");
  clParser.addOption("fitFiles", {"-f", "--file"}, "Specify path to fitter output files", -1);
  clParser.addOption("verbose", {"-v", "--verbose"}, "Set the verbosity level", 1, true);
  clParser.addOption("showCorrelationsWith", {"--show-correlations-with"}, "Show all correlation coefficients of a given par wrt others", -1);
  clParser.addOption("outFolder", {"-o", "--out-folder"}, "Set output folder where files will be writen", 1);

  clParser.addDummyOption("Triggers");
  clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Don't write files on disk");
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
          GenericToolbox::getFolderPathFromFilePath(file).empty() ? "./" : GenericToolbox::getFolderPathFromFilePath(file)
        ),
        GenericToolbox::getFileNameFromFilePath(file, false)
    )};

    if( not clParser.isOptionTriggered("dryRun") ){
      LogWarning << "Output files will be writen under: " << outDir << std::endl;
    }

    auto f = std::make_shared<TFile>(file.c_str());
    LogContinueIf(not GenericToolbox::doesTFileIsValid(f.get()), "Could not open \"" << file << "\"");
    LogContinueIf(f->Get("gundamFitter") == nullptr, "Not a gundam fitter output file.");

    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "commandLine_TNamed"), [](TNamed* obj_){
      LogInfo << blueLightText << "Cmd line: " << resetColor << obj_->GetTitle() << std::endl;
    });
    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "gundamVersion_TNamed"), [](TNamed* obj_){
      LogInfo << blueLightText << "Ran with GUNDAM version: " << resetColor << obj_->GetTitle() << std::endl;
    });
    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "unfoldedConfig_TNamed"), [&](TNamed* obj_){
      if( not clParser.isOptionTriggered("dryRun") ){
        if( not GenericToolbox::doesPathIsFolder(outDir) ){ GenericToolbox::mkdirPath(outDir); }
        auto outConfigPath = GenericToolbox::joinPath(outDir, "config.json");
        LogInfo << blueLightText << "Writing unfolded config under: " << resetColor << outConfigPath << std::endl;
        GenericToolbox::dumpStringInFile( outConfigPath, obj_->GetTitle() );
      }
    });


    // FitterEngine/propagator
    do {
      auto pathPropagator{GenericToolbox::joinPath("FitterEngine", "propagator")};
      if( not readObject(f.get(), pathPropagator) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPropagator << resetColor << std::endl;
      LogScopeIndent;

      readObject<TMatrixT<double>>(f.get(), GenericToolbox::joinPath(pathPropagator, "globalCovarianceMatrix_TMatrixD"), [&](TMatrixT<double>* matrix){
        readMatrix("Global prior covariance matrix", matrix);
      });

      for( auto& parSet : GenericToolbox::lsSubDirTDirectory( f->Get<TDirectory>(pathPropagator.c_str()) ) ){

        if( f->Get(GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixD").c_str()) ){
          readObject<TMatrixT<double>>(
              f.get(),
              GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixD"),
              [&](TMatrixT<double>* matrix){
                readMatrix(parSet + " prior covariance matrix", matrix);
              });
        }
        else if( f->Get(GenericToolbox::joinPath(pathPropagator, parSet, "covarianceMatrix_TMatrixDSym").c_str()) ){
          readObject<TMatrixTSym<double>>(
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
      if( not readObject(f.get(), pathPreFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPreFit << resetColor << std::endl;
      LogScopeIndent;

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "llhState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.isOptionTriggered("verbose") ){
          LogInfo << blueLightText << "Pre-fit Likelihood state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( not clParser.isOptionTriggered("dryRun") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
          if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
          LogInfo << blueLightText << "Writing pre-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "parState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.getOptionVal("verbose", 0) >= 1 ){
          LogInfo << blueLightText << "Pre-fit parameters state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( not clParser.isOptionTriggered("dryRun") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
          if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".json");
          LogInfo << blueLightText << "Writing pre-fit LLH parameter injector under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });

    } while( false ); // allows to skip if not found

    /// FitterEngine/postFit
    do {
      auto pathPostFit{GenericToolbox::joinPath("FitterEngine", "postFit")};
      if( not readObject(f.get(), pathPostFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPostFit << resetColor << std::endl;
      LogScopeIndent;

      std::string minimizationAlgo{};
      readObject<TTree>(f.get(), GenericToolbox::joinPath(pathPostFit, "bestFitStats"), [&](TTree* tree){
        tree->GetEntry(0);

        bool converged{tree->GetLeaf("fitConverged")->GetValue() == 1};
        LogInfo << (converged? greenLightText: redLightText) << "Did the fit converge? " << (converged ? "yes": "no") << resetColor << std::endl;

        int statusCode{int(tree->GetLeaf("fitStatusCode")->GetValue())};
        LogInfo << blueLightText << "Fit status code: " << resetColor;
        LogInfo << ( GenericToolbox::doesKeyIsInMap( statusCode, GundamUtils::minuitStatusCodeStr ) ? GundamUtils::minuitStatusCodeStr.at(statusCode) : std::to_string(statusCode) );
        LogInfo << std::endl;
      });
      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "llhState_TNamed"), [&](TNamed* injectorStr){
        LogInfo << blueLightText << "Post-fit Likelihood state: " << resetColor << injectorStr->GetTitle() << std::endl;
        if( not clParser.isOptionTriggered("dryRun") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
          if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
          auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
          LogInfo << blueLightText << "Writing post-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
          GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
        }
      });
      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "parState_TNamed"), [&](TNamed* injectorStr){
        if( clParser.getOptionVal("verbose", 0) >= 1 ){
          LogInfo << blueLightText << "Post-fit parameters state: " << resetColor << injectorStr->GetTitle() << std::endl;
        }
        if( not clParser.isOptionTriggered("dryRun") ){
          auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
          if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
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
          readObject<TNamed>( f.get(), GenericToolbox::joinPath( parSetPath, parEntry, "fullTitle_TNamed" ), [&](TNamed* obj){
            bool isEnabled{false};
            readObject<TNamed>( f.get(), GenericToolbox::joinPath( parSetPath, parEntry, "isEnabled_TNamed" ), [&](TNamed* obj){
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

      readObject<TDirectory>( f.get(), pathPostFit, [&](TDirectory* postFitDir_){
        readObject<TDirectory>( postFitDir_, {{"Hesse"}, {"Migrad"}}, [&](TDirectory* hesseDir_){
          readObject<TH2D>( hesseDir_, "hessian/postfitCorrelationOriginal_TH2D", [&](TH2D* cor_){

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

            std::vector<std::pair<std::string, double>> corrDict{};
            corrDict.reserve(cor_->GetNbinsX()-1);
            for( int iPar = 1 ; iPar <= cor_->GetNbinsX() ; iPar++ ){
              corrDict.emplace_back(cor_->GetXaxis()->GetBinLabel(iPar), cor_->GetBinContent(selectedParIndex, iPar));
            }

            GenericToolbox::sortVector(corrDict, [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b){
              return std::abs( a.second ) > std::abs( b.second );
            });

            GenericToolbox::TablePrinter t;
            t << "Fit parameter" << GenericToolbox::TablePrinter::NextColumn;
            t << "Correlation" << GenericToolbox::TablePrinter::NextLine;

            for( auto& corrEntry : corrDict ){
              t << corrEntry.first << GenericToolbox::TablePrinter::NextColumn;
              t << corrEntry.second * 100 << " %" << GenericToolbox::TablePrinter::NextLine;
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

template<typename T> bool readObject( TDirectory* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ ){
  T* obj;
  for( auto& objPath : objPathList_ ){
    obj = f_->Get<T>(objPath.c_str());
    if( obj != nullptr ){ break; }
  }
  if( obj == nullptr ){
    LogErrorIf(not quiet) << redLightText << "Could not find object among names: " << resetColor << GenericToolbox::parseVectorAsString(objPathList_) << std::endl;
    return false;
  }
  action_(obj);
  return true;
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
