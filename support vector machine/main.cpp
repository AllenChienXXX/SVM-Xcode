#include <iostream>
#include "preprocess.hpp"
#include"SVM.hpp"


int main(int argc, char** argv)
{
//    preprocess data("/Users/allenchien/Documents/program/C++/support vector machine/Datasets/Breast Cancer Wisconsin (Diagnostic) Data Set.csv");
    
    vector<vector<double>> inputs = { {-3,-1},{-4,-2},{-2,-3},{-1,-4},{5,5},{3,5}, {2,7},{7,4} };
    vector<double> labels = { -1,-1,-1,-1,1,1,1,1 };
    SVM mysvm(inputs, labels);

    for (int i = 0; i < 1000; i++) {
        mysvm.train();
    }
//    mysvm.get_weights();
//    mysvm.print_v(mysvm.weights);

    mysvm.predict({ {1,2},{7,1},{8.2,1},{1,3} });

    mysvm.get_weights();
    //mysvm.print_alpha();
    mysvm.plot(argc, argv);
}

