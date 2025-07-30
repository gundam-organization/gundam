

void parseFormulaFromTree(){

  string formula = "a + b";
  // open tfiel and ttree
  TFile* file = new TFile("treeToParse.root");
  TTree* tree = (TTree*)file->Get("tTree");
  // fill a hsitogram with the formula
  TH1F* h = new TH1F("h", "Formula distribution", 10, 0, 10);
  // convert the formula from string to TFormula
  TFormula* f = new TFormula("f", formula.c_str());
// create a buffer to store the values of the branches
  double x = 0;
  double y = 0;
  // set the branches
  tree->SetBranchAddress("x", &x);
  tree->SetBranchAddress("y", &y);
  // loop over the tree
  for(int i = 0; i<tree->GetEntries(); i++){
    tree->GetEntry(i);
    // evaluate the formula
    double value = f->Eval(x,y);
    // fill the histogram
    cout<<value<<endl;
    h->Fill(value);
  }





}