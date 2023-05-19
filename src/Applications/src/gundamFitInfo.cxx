//
// Created by Adrien Blanchet on 17/05/2023.
//

#include "VersionConfig.h"
#include "ConfigUtils.h"
#include "GlobalVariables.h"
#include "GundamGreetings.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

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
template<typename T> bool readObject( TFile* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ = [](T*){} );
template<typename T> bool readObject( TFile* f_, const std::string& objPath_, const std::function<void(T*)>& action_ = [](T*){} ){ return readObject(f_, std::vector<std::string>{objPath_}, action_); }
bool readObject( TFile* f_, const std::string& objPath_){ return readObject<TObject>(f_, objPath_); }
void readMatrix( const std::string& title_, TMatrixD* matrix_ );



int main(int argc, char** argv){

  // --------------------------
  // Greetings:
  // --------------------------
  GundamGreetings g;
  g.setAppName("GundamFitInfo");
  g.hello();

  // --------------------------
  // Read Command Line Args:
  // --------------------------

  clParser.getDescription() << "gundamFitInfo is a program that reads in the output files of gundamFitter." << std::endl;

  LogInfo << clParser.getDescription().str() << std::endl;

  clParser.addDummyOption("Options");
  clParser.addOption("fitFiles", {"-f", "--file"}, "Specify path to fitter output files", -1);
  clParser.addOption("verbose", {"-v", "--verbose"}, "Set the verbosity level", 1, true);

  clParser.addDummyOption("Triggers");
  clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Don't write files on disk");

  clParser.addDummyOption();

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;


  LogThrowIf( clParser.getNbValueSet("fitFiles") == 0, "no fit files provided" );


  if( not clParser.isOptionTriggered("dryRun") ){
    LogWarning << "Output files will be writen under: " << GenericToolbox::getCurrentWorkingDirectory() << std::endl;
  }
  for( auto& file : clParser.getOptionValList<std::string>("fitFiles") ){
    LogInfo << std::endl << magentaLightText << "Opening: " << resetColor << file << std::endl;
    LogScopeIndent;

    auto f = std::make_shared<TFile>(file.c_str());
    LogContinueIf(not GenericToolbox::doesTFileIsValid(f.get()), "Could not open \"" << file << "\"");
    LogContinueIf(f->Get("gundamFitter") == nullptr, "Not a gundam fitter output file.");

    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "commandLine_TNamed"), [](TNamed* obj_){
      LogInfo << blueLightText << "Cmd line: " << resetColor << obj_->GetTitle() << std::endl;
    });
    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "gundamVersion_TNamed"), [](TNamed* obj_){
      LogInfo << blueLightText << "Ran with GUNDAM version: " << resetColor << obj_->GetTitle() << std::endl;
    });


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


    if( clParser.isOptionTriggered("dryRun") ){
      LogAlert << "Dry run set. Not doing actions involving writing of files on disk" << std::endl;
      continue;
    }

    auto outDir{GenericToolbox::getFileNameFromFilePath(file, false)};


    readObject<TNamed>(f.get(), GenericToolbox::joinPath("gundamFitter", "unfoldedConfig_TNamed"), [&](TNamed* obj_){
      if( not GenericToolbox::doesPathIsFolder(outDir) ){ GenericToolbox::mkdirPath(outDir); }
      auto outConfigPath = GenericToolbox::joinPath(outDir, "config.json");
      LogInfo << blueLightText << "Writing unfolded config under: " << resetColor << outConfigPath << std::endl;
      GenericToolbox::dumpStringInFile( outConfigPath, obj_->GetTitle() );
    });


    /// Pre-fit folder
    do {
      auto pathPreFit{GenericToolbox::joinPath("FitterEngine", "preFit")};
      if( not readObject(f.get(), pathPreFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPreFit << resetColor << std::endl;
      LogScopeIndent;

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "preFitLlhState_TNamed"), [&](TNamed* injectorStr){
        auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
        if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
        auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
        LogInfo << blueLightText << "Writing pre-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
        GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
      });

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPreFit, "preFitParState_TNamed"), [&](TNamed* injectorStr){
        auto outSubDir{GenericToolbox::joinPath( outDir, pathPreFit)};
        if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
        auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".json");
        LogInfo << blueLightText << "Writing pre-fit LLH parameter injector under: " << resetColor << outConfigPath << std::endl;
        GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
      });

    } while( false ); // allows to skip if not found


    /// Post-fit folder
    do {
      auto pathPostFit{GenericToolbox::joinPath("FitterEngine", "postFit")};
      if( not readObject(f.get(), pathPostFit) ){ break; }

      LogInfo << cyanLightText << "Reading inside: " << pathPostFit << resetColor << std::endl;
      LogScopeIndent;

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "postFitLlhState_TNamed"), [&](TNamed* injectorStr){
        auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
        if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
        auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".txt");
        LogInfo << blueLightText << "Writing post-fit LLH stats under: " << resetColor << outConfigPath << std::endl;
        GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
      });

      readObject<TNamed>(f.get(), GenericToolbox::joinPath(pathPostFit, "postFitParState_TNamed"), [&](TNamed* injectorStr){
        auto outSubDir{GenericToolbox::joinPath( outDir, pathPostFit)};
        if( not GenericToolbox::doesPathIsFolder( outSubDir ) ){ GenericToolbox::mkdirPath( outSubDir ); }
        auto outConfigPath = GenericToolbox::joinPath( outSubDir, std::string(injectorStr->GetName()) + ".json");
        LogInfo << blueLightText << "Writing post-fit LLH parameter injector under: " << resetColor << outConfigPath << std::endl;
        GenericToolbox::dumpStringInFile( outConfigPath, injectorStr->GetTitle() );
      });

    } while( false ); // allows to skip if not found

    f->Close();
  }

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  return EXIT_SUCCESS;
}

template<typename T> bool readObject( TFile* f_, const std::vector<std::string>& objPathList_, const std::function<void(T*)>& action_ ){
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
