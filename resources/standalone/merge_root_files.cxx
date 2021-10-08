void merge_root_files(const std::string& file_list, const std::string& output_file, const unsigned long max_tree_size = 200000000000)
{
    //This code uses TFileMerger to merge a list of ROOT files.
    //file_list : a text file containing the ROOT files to merge, one per line
    //output_file : name of the merged output file
    //max_tree_size : the new max size for TTree, given in bytes

    //Set a new max tree size in bytes. The default function value above
    //sets the new max size to 200 GB.
    std::cout << "Max tree size : " << TTree::GetMaxTreeSize() << std::endl;
    TTree::SetMaxTreeSize(max_tree_size);
    std::cout << "Updated tree size : " << TTree::GetMaxTreeSize() << std::endl;

    //Get the file names from the text file specified by file_list
    std::vector<std::string> v_file_names;
    std::ifstream fin(file_list, std::ios::in);
    if(!fin.is_open())
    {
        std::cout << "Failed to open file : " << file_list << std::endl;
        return;
    }
    else
    {
        std::string line;
        while(std::getline(fin, line))
        {
            //If the first character on the line is '#',
            //then ignore that line.
            if(line.front() != std::string("#"))
                v_file_names.emplace_back(line);
        }
    }
    std::cout << "Saving to " << output_file << std::endl;

    TFileMerger fm;
    fm.SetPrintLevel(3);
    fm.SetFastMethod();
    fm.OutputFile(output_file.c_str(), "RECREATE");

    for(int i = 0; i < v_file_names.size(); ++i)
    {
        std::cout << "Adding " << v_file_names.at(i) << " to merge." << std::endl;
        fm.AddFile(v_file_names.at(i).c_str());
    }

    std::cout << "Performing merge..." << std::endl;
    bool success = fm.Merge();

    if(success)
        std::cout << "Merge successful." << std::endl;
    else
        std::cout << "Something went wrong." << std::endl;

    return;
}
