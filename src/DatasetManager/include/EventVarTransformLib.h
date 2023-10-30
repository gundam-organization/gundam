//
// Created by Adrien BLANCHET on 11/11/2022.
//

#ifndef GUNDAM_EVENTVARTRANSFORMLIB_H
#define GUNDAM_EVENTVARTRANSFORMLIB_H

#include "EventVarTransform.h"

class EventVarTransformLib : public EventVarTransform{

public:
  EventVarTransformLib() = default;
  explicit EventVarTransformLib(const nlohmann::json& config_){ this->readConfig(config_); }

  void reload();

protected:
  void initializeImpl() override;
  void readConfigImpl() override;

  void loadLibrary();
  void initInputFormulas();


  double evalTransformation(const PhysicsEvent& event_, std::vector<double>& inputBuffer_) const override;

private:
  std::string _libraryFile_{};
  void* _loadedLibrary_{nullptr};
  void* _evalVariable_{nullptr};

};


#endif //GUNDAM_EVENTVARTRANSFORMLIB_H
