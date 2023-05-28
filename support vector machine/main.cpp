#include <iostream>
#include "preprocess.hpp"
#include"SVM.hpp"


int main(int argc, char** argv)
{
    preprocess data("/Users/allenchien/Documents/program/C++/support vector machine/Datasets/Breast Cancer Wisconsin (Diagnostic) Data Set.csv");
    data.print_label();
    
//    vector<vector<double>> inputs = { {-30,-10},{-40,-20},{-20,-30},{-10,-40},{50,50},{30,50}, {20,70},{70,40} };
//    vector<vector<double>> inputs = { {10,10},{10,20},{-10,30},{10,40},{20,10},{20,20}, {20,30},{20,40} };
//    vector<vector<double>> inputs = { {33,-33},{10,-20},{10,-30},{20,-40},{20,10},{20,20}, {20,30},{20,40} };


//    vector<double> labels = { -1,-1,-1,-1,1,1,1,1 };
//    SVM mysvm(inputs, labels
    
//    mysvm.train(10000);

//    mysvm.predict({ {10,20},{70,10},{82,10},{10,30},{70,40} });

//    mysvm.get_weights();
//    mysvm.print_alpha();
//    mysvm.plot(argc, argv);
}

