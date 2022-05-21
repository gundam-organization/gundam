//
// Created by Adrien BLANCHET on 18/05/2022.
//

#ifndef GUNDAM_TREEEVENTBUFFER_H
#define GUNDAM_TREEEVENTBUFFER_H

#include "GenericToolbox.Root.LeafHolder.h"

#include "TTree.h"

#include "vector"
#include "string"

class TreeEventBuffer {

public:
  TreeEventBuffer();
  virtual ~TreeEventBuffer();

  void setLeafNameList(const std::vector<std::string> &leafNameList);

  void hook(TTree* tree_);

  const std::vector<GenericToolbox::LeafHolder> &getLeafContentList() const;

  int fetchLeafIndex(const std::string& leafName_) const;
  const GenericToolbox::LeafHolder& getLeafContent(const std::string& leafName_) const;
  std::string getSummary();

private:
  std::vector<std::string> _leafNameList_;
  std::vector<GenericToolbox::LeafHolder> _leafContentList_;

};


#endif //GUNDAM_TREEEVENTBUFFER_H
