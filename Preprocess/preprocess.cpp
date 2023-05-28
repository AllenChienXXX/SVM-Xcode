//
//  preprocess.cpp
//  support vector machine
//
//  Created by Allen Chien on 5/24/23.
//

#include "preprocess.hpp"

preprocess::preprocess(){
    
}

preprocess::preprocess(string filename){
    ifstream file(filename);
    bool lab = true;
    
    string line;
    vector<vector<double>> columns;

    while (getline(file, line)) {
        istringstream iss(line);
        string value;
        vector<double> row;
        if(lab){
            while(getline(iss, value, ',')) {
                label.push_back(value);
            }
            lab = false;
            continue;
        }
        try{
            while (getline(iss, value, ',')) {
                row.push_back(stod(value));
            }
            columns.push_back(row);

        }catch (const std::exception& e) {
            cout << "Possible data loss: "<<value << e.what() << endl;
        }
        
    }
}

void preprocess::print_label(){
    for(int i=0;i<label.size();i++){
        printf("%s\n",label[i].c_str());
    }
}
