#include "DatasetTreeChain.h"

#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <TDirectory.h>
#include <TFile.h>

#include <algorithm>
#include <utility>

DatasetTreeChain::~DatasetTreeChain(){
  TChain::SetNotify(nullptr);
  TChain::ResetBranchAddresses();
  clearFriends();
}

void DatasetTreeChain::addFile(const std::string& filePath_, const std::string& treePath_,
                               Long64_t nbEntries_, std::vector<FriendTree> friends_, Long64_t nbEntriesToRead_){
  if( nbEntriesToRead_ < 0 ){ nbEntriesToRead_ = nbEntries_; }
  LogThrowIf(nbEntriesToRead_ > nbEntries_, "Requested more entries than available in " << filePath_);
  // AddFile(0) asks ROOT to read the full tree; it does not mean zero entries.
  if( nbEntriesToRead_ == 0 ){ return; }
  LogThrowIf(TChain::AddFile(filePath_.c_str(), nbEntries_, treePath_.c_str()) == 0,
             "Could not add input tree " << filePath_ << ":" << treePath_);
  _fileEntryList_.push_back({filePath_, std::move(friends_), _nbEntries_, _nbPhysicalEntries_});
  _nbEntries_ += nbEntriesToRead_;
  _nbPhysicalEntries_ += nbEntries_;
}

Long64_t DatasetTreeChain::LoadTree(Long64_t entry_){
  if( entry_ < 0 or entry_ >= _nbEntries_ ){
    _readEntry_ = -1;
    return -2;
  }
  // ROOT replaces advertised file sizes with the actual tree sizes on open.
  // Map the retained prefixes (fractionOfEntries) to full-chain entry numbers.
  auto source = std::upper_bound(_fileEntryList_.begin(), _fileEntryList_.end(), entry_,
      [](Long64_t entry, const FileEntry& file){ return entry < file.beginEntry; });
  --source;
  const Long64_t physicalEntry = source->physicalBeginEntry + entry_ - source->beginEntry;

  const int previousTreeNumber = GetTreeNumber();
  if( GetTree() != nullptr and (physicalEntry < GetTreeOffset()[previousTreeNumber]
      or physicalEntry >= GetTreeOffset()[previousTreeNumber + 1]) ){
    // Close the previous friends before ROOT opens the next main file, so
    // transitions do not retain two complete groups of input files.
    clearFriends();
  }

  auto* notifier = GetNotify();
  Long64_t localEntry;
  bool changedTree{false};
  {
    // TreeBuffer must see the new friends when it refreshes branch addresses
    // and TTreeFormula leaves. Defer ROOT's notification until they are attached.
    GenericToolbox::ScopedGuard notifyGuard{
      [&]{ TChain::SetNotify(nullptr); },
      [&]{ TChain::SetNotify(notifier); }
    };
    localEntry = TChain::LoadTree(physicalEntry);
    _readEntry_ = localEntry < 0 ? -1 : entry_;
    if( localEntry < 0 ){ return localEntry; }
    changedTree = previousTreeNumber != GetTreeNumber() or not _friendsLoaded_;
    if( changedTree ){
      loadFriends();
      _friendsLoaded_ = true;
      // The main tree was positioned before its friends were attached. Set
      // their local cursors as well, including when a reader starts mid-file.
      GetTree()->LoadTree(localEntry);
    }
  }
  if( changedTree and notifier != nullptr and not notifier->Notify() ){ return -6; }
  return localEntry;
}

void DatasetTreeChain::SetBranchStatus(const char* name_, Bool_t enabled_, UInt_t* found_){
  if( GetTree() == nullptr and LoadTree(0) < 0 ){ return; }
  // TreeBuffer's notifier reapplies statuses after attaching each new group
  // of friends. Persisting friend branch names in TChain's status list would
  // make ROOT look for them before the next friends have been attached.
  GetTree()->SetBranchStatus(name_, enabled_, found_);
}

TList* DatasetTreeChain::GetListOfFriends() const{
  return GetTree() == nullptr ? TChain::GetListOfFriends() : GetTree()->GetListOfFriends();
}

TTree* DatasetTreeChain::GetFriend(const char* name_) const{
  return GetTree() == nullptr ? nullptr : GetTree()->GetFriend(name_);
}

const char* DatasetTreeChain::GetFriendAlias(TTree* tree_) const{
  return GetTree() == nullptr ? nullptr : GetTree()->GetFriendAlias(tree_);
}

void DatasetTreeChain::clearFriends(){
  if( auto* tree = GetTree() ){
    for( auto* friendTree : _friendTreeList_ ){
      tree->RemoveFriend(friendTree);
    }
  }
  _friendTreeList_.clear();
  _friendFileMap_.clear();
  _friendsLoaded_ = false;
}

void DatasetTreeChain::loadFriends(){
  const auto& source = _fileEntryList_.at(GetTreeNumber());
  for( const auto& friendSource : source.friends ){
    TFile* file = TChain::GetFile();
    if( friendSource.filePath != source.filePath ){
      auto& ownedFile = _friendFileMap_[friendSource.filePath];
      if( ownedFile == nullptr ){
        TDirectory::TContext context;
        ownedFile.reset(TFile::Open(friendSource.filePath.c_str(), "READ"));
      }
      file = ownedFile.get();
    }
    LogThrowIf(file == nullptr or file->IsZombie(),
               "Could not open friend file " << friendSource.filePath
               << ". Check the input path and the open-file limit (ulimit -n).");
    auto* tree = file->Get<TTree>(friendSource.treePath.c_str());
    LogThrowIf(tree == nullptr, "Missing friend tree " << friendSource.treePath << " in " << friendSource.filePath);
    LogThrowIf(tree == GetTree(), "The main tree cannot be its own friend: " << friendSource.name);
    LogThrowIf(tree->GetEntries() != GetTree()->GetEntries(),
               "Friend \"" << friendSource.name << "\" is not entry-aligned with its main tree in " << source.filePath);
    LogThrowIf(GetTree()->AddFriend(tree, friendSource.name.c_str()) == nullptr,
               "Could not attach friend tree " << friendSource.name);
    _friendTreeList_.push_back(tree);
  }
}
