#include <iostream>
#include "preprocess.hpp"
#include"SVM.hpp"


int main(int argc, char** argv)
{
//    preprocess file("/Users/allenchien/Documents/program/C++/support vector machine/Datasets/Breast Cancer Wisconsin (Diagnostic) Data Set.csv");
    preprocess file("/Users/allenchien/Documents/program/C++/support vector machine/Datasets/data.csv");
//    file.print_label();
//    file.show_data();
    vector<vector<double>> X = file.read_column(1, 2);
    vector<double> Y = file.read_column(0);

    
//    vector<vector<double>> X = { {-30,-10},{-40,-20},{-20,-30},{-10,-40},{50,50},{30,50}, {20,70},{70,40} };
//    vector<vector<double>> X = { {10,10},{10,20},{-10,30},{10,40},{20,10},{20,20}, {20,30},{20,40} };
//    vector<vector<double>> X = { {33,-33},{10,-20},{10,-30},{20,-40},{20,10},{20,20}, {20,30},{20,40} };


//    vector<double> Y = { -1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0 };
    SVM mysvm(X,Y);
    
    mysvm.train(150);
//    mysvm.print_v(mysvm.outputs);
//    mysvm.print_v(mysvm.errors);


//    mysvm.predict({ {10,20},{70,10},{82,10},{10,30},{70,40} });

//    mysvm.get_weights();
//    mysvm.print_alpha();
//    mysvm.plot(argc, argv);
}
