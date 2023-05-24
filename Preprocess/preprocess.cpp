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
    
    string line;
    vector<vector<string>> columns;

    while (getline(file, line)) {
       istringstream iss(line);
       string value;
       vector<std::string> row;

       while (getline(iss, value, ',')) {
           row.push_back(value);
       }
       // Store each column in the vector
       for (size_t i = 0; i < row.size(); i++) {
            if (columns.size() <= i) {
               columns.push_back(vector<string>());
            }
            columns[i].push_back(row[i]);
       }
    }
    // Print the stored columns
    for (const auto& column : columns) {
        for (const auto& value : column) {
           cout << value << " ";
        }
        cout << endl;
    }
}

