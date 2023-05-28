//
//  preprocess.hpp
//  support vector machine
//
//  Created by Allen Chien on 5/24/23.
//

#ifndef preprocess_hpp
#define preprocess_hpp

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#endif /* preprocess_hpp */

using namespace std;

class preprocess{
public:
    preprocess();
    
    preprocess(string filename);
    
    vector<double> read_column();
    
    void stdscaler();
    
    void print_label();
    
private:
    vector<double> data;
    vector<string> label;
};

