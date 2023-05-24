#include <iostream>
#include"SVM.hpp"


int main(int argc, char** argv)
{
    vector<vector<double>> inputs = { {1,1},{1,2},{1,3},{1,4},{2,1},{2,2}, {2,3},{2,4} };
    vector<double> labels = { -1,-1,-1,-1,1,1,1,1 };
    SVM mysvm(inputs, labels);

    for (int i = 0; i < 200; i++) {
        mysvm.train();
        //mysvm.print_alpha();
        //mysvm.print_v(mysvm.outputs);
        //mysvm.print_v(mysvm.errors);
    }
    //mysvm.get_weights();
    //mysvm.print_v(mysvm.weights);

    //mysvm.predict({ {1,2},{7,1},{8.2,1},{1,3} });

    mysvm.get_weights();
    //mysvm.print_alpha();
    mysvm.plot(argc, argv);
}

