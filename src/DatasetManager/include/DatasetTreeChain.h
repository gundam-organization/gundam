#ifndef GUNDAM_DATASET_TREE_CHAIN_H
#define GUNDAM_DATASET_TREE_CHAIN_H

#include <TChain.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// A reader owns one main chain and shares each current friend file between
// its trees. Instances, files and tree buffers must not be shared by threads.
class DatasetTreeChain : public TChain {
public:
  struct FriendTree{
    std::string name;
    std::string filePath;
    std::string treePath;
  };

  ~DatasetTreeChain() override;
  void addFile(const std::string& filePath_, const std::string& treePath_,
               Long64_t nbEntries_, std::vector<FriendTree> friends_, Long64_t nbEntriesToRead_ = -1);

  using TChain::GetEntries;
  Long64_t GetEntries() const override { return _nbEntries_; }
  Long64_t GetReadEntry() const override { return _readEntry_; }
  Long64_t LoadTree(Long64_t entry_) override;
  void SetBranchStatus(const char* name_, Bool_t enabled_ = true, UInt_t* found_ = nullptr) override;
  TList* GetListOfFriends() const override;
  TTree* GetFriend(const char* name_) const override;
  const char* GetFriendAlias(TTree* tree_) const override;

private:
  struct FileEntry{
    std::string filePath;
    std::vector<FriendTree> friends;
    Long64_t beginEntry;
    Long64_t physicalBeginEntry;
  };

  void clearFriends();
  void loadFriends();

  std::vector<FileEntry> _fileEntryList_;
  std::map<std::string, std::unique_ptr<TFile>> _friendFileMap_;
  std::vector<TTree*> _friendTreeList_;
  bool _friendsLoaded_{false};
  Long64_t _nbEntries_{0};
  Long64_t _nbPhysicalEntries_{0};
  Long64_t _readEntry_{-1};
};

#endif
