//
// Created by Nadrino on 07/03/2025.
//

#ifndef DIALMANAGER_H
#define DIALMANAGER_H

#include "ParametersManager.h"
#include "DialCollection.h"

#include "ConfigUtils.h"

#include <vector>


class DialManager : public JsonBaseClass {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  DialManager() = default;

  // setters
  void setParametersManager(ParametersManager* parameterManagerPtr_){ _parametersManagerPtr_ = parameterManagerPtr_; }

  // const getters
  [[nodiscard]] auto& getDialCollectionList() const{ return _dialCollectionList_; }

  // mutable getters
  auto& getDialCollectionList(){ return _dialCollectionList_; }

  // main methods
  void shrinkDialContainers();
  void clearEventByEventDials();
  void invalidateInputBuffers();
  void updateDialState();
  void printSummaryTable() const;

private:
  ParametersManager* _parametersManagerPtr_{nullptr};
  std::vector<DialCollection> _dialCollectionList_;
};



#endif //DIALMANAGER_H
