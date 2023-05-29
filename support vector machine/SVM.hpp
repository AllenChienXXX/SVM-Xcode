#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include <GLUT/GLUT.h>
#include <OpenGL/OpenGL.h>

using namespace std;

class SVM
{
public:
    SVM(vector<vector<double>>inputs, vector<double>labels, double C = 200, double tolerence = 0.0001,double threshold = 10.0);

    void init_alpha_b();

    void get_output();

    void get_error();

    void train(int epoch = 10);

    void update();

    //check lowerbound and upperbound
    void check_condition(int index1, int index2, double &Lbound, double &Hbound);

    //returns the new alpha
    double update_alpha(int index1, int index2);

    //adjust alpha with lower and upperbound
    void adjust_alpha(double Lbound, double Hbound);

    void update_bias(int index1, int index2);

    int select_alpha1();

    int select_alpha2(int inde1R);

    vector<double> v_product(vector<double> v1, vector<double> v2);

    double Kernel_function(vector<double> x1, vector<double> x2);

    void print_alpha();

    void print_v(vector<double> v);

    void get_weights();

    void predict(vector<vector<double>> vec);

    void plot(int argc, char** argv);

    void static display();

    void static display_points();
    
    double get_accuracy();

private:
    
    static vector<vector<double>> inputs;
    static vector<double> labels;
    vector<double> alpha;
    vector<double> outputs;
    vector<double> errors;
    vector<int> updatelist;
    static vector<double> weights;
    
    unsigned int len;

    //range C
    double C;
    //tolerence
    double tol;

    double static bias;
    
    double static thresh;

    double alpha1_old;
    double alpha1_new;
    double alpha2_old;
    double alpha2_new;
};
