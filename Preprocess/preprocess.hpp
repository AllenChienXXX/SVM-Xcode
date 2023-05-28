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
    
    vector<vector<double>> read_column(int i1,int i2);
    
    vector<double> read_column(int i1);
    
    vector<string> read_label(int i1,int i2);
    
    void stdscaler(int index);
    
    void calculate_m_std(int index);
    
    void print_label();
    
    void show_data();
    
private:
    vector<string> label;
    vector<vector<double>> raw_data;
    
    double mean;
    double std;

};

