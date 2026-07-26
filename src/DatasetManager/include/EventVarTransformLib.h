//
// Created by Adrien BLANCHET on 11/11/2022.
//

#ifndef GUNDAM_EVENTVARTRANSFORMLIB_H
#define GUNDAM_EVENTVARTRANSFORMLIB_H

#include "EventVarTransform.h"

class EventVarTransformLib : public EventVarTransform {

public:
  EventVarTransformLib() = default;
  explicit EventVarTransformLib(const ConfigReader& config_){ this->configure(config_); }

  void configureFromVariableDict(const std::string& outputVariableName_, ConfigReader& config_);
  void reload();

  void setLibraryFile(const std::string& libraryFile_){ _libraryFile_ = libraryFile_; }
  const std::string& getLibraryFile() const{ return _libraryFile_; }

protected:
  void initializeImpl() override;
  void configureImpl() override;

  void loadLibrary();
  void initInputFormulas();

  double evalTransformation( const Event& event_, std::vector<double>& inputBuffer_) const override;

private:
  std::string _libraryFile_{};
  void* _loadedLibrary_{nullptr};
  void* _evalVariable_{nullptr};

};


#endif //GUNDAM_EVENTVARTRANSFORMLIB_H
