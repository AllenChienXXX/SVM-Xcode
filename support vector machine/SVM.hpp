#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>


using namespace std;

class SVM
{
public:
    SVM(vector<vector<float>>inputs, vector<float>labels, float C = 200, float tolerence = 0.0001);

    void init_alpha_b(int len);

    void get_output();

    void get_error();

    void train(int epoch = 10);

    void update();

    //check lowerbound and upperbound
    void check_condition(int index1, int index2, float &Lbound, float &Hbound);

    //returns the new alpha
    float update_alpha(int index1, int index2);

    //adjust alpha with lower and upperbound
    void adjust_alpha(float Lbound, float Hbound);

    void update_bias(int index1, int index2);

    int select_alpha1();

    int select_alpha2(int inde1R);

    vector<float> v_product(vector<float> v1, vector<float> v2);

    float Kernel_function(vector<float> x1, vector<float> x2);

    void print_alpha();

    void print_v(vector<float> v);

    void get_weights();

    void predict(vector<vector<float>> vec);



//private:
    
    vector<vector<float>> inputs;
    vector<float> labels;
    vector<float> alpha;
    vector<float> outputs;
    vector<float> errors;
    vector<int> updatelist;
    vector<float> weights;

    //range C
    float C;
    //tolerence
    float tol;

    float bias;

    float alpha1_old;
    float alpha1_new;
    float alpha2_old;
    float alpha2_new;
};
