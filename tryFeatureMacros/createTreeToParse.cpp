void createTreeToParse(){
    // This root macro creates a tree with a branch with a formula
    // It gets as argument a formula and creates a tree with a branch with the formula
    // run it with root -l createTreeToParse.cpp
    TFile* file = new TFile("treeToParse.root", "RECREATE");
    TTree* tree = new TTree("tTree", "Tree with a formula");
    double x = 0;
    double y = 0;
    tree->Branch("x", &x);
    tree->Branch("y", &y);
    for(int i = 0; i<10; i++){
        x = (i+1);
        y = 2*(i+1);
        tree->Fill();
    }
    tree->Print();
    tree->Write();
    delete tree;
}