#include "SVM.hpp"


SVM::SVM(vector<vector<float>>inputs, vector<float>labels, float C, float tolerence) {
    this->inputs = inputs;
    this->labels = labels;
    this->C = C;
    this->tol = tolerence;
    int len = inputs.size();
    init_alpha_b(len);
}

void SVM::init_alpha_b(int len) {
    this->bias = 0.0;
    alpha = vector<float>(len, 0.0);
    outputs = vector<float>(len, 0.0);
    errors = vector<float>(len, 0.0);
    weights = vector<float>(inputs[0].size(), 0.0);
}

//sumof(i=1,n)(alpha_i*y_i*K_i_x) = w T x +b
void SVM::get_output() {
    for (int i = 0; i < outputs.size(); i++) {
        float sum = 0.0;
        for (int j = 0; j < outputs.size(); j++) {
            sum += alpha[j] * labels[j] * Kernel_function(inputs[i], inputs[j]);
        }
        sum += bias;
        outputs[i] = sum;
    }
}

//E_i = Output_i - labels_i
void SVM::get_error() {
    for (int i = 0; i < errors.size(); i++) {
        errors[i] = outputs[i] - labels[i];
    }
}

void SVM::train(int epoch) {
    get_output();
    get_error();

    update();
}

void SVM::update() {
    //get index for alpha1
    int index1 = select_alpha1();
    //get index for alpha2
    int index2 = select_alpha2(index1);
    alpha1_old = alpha[index1];
    alpha2_old = alpha[index2];
    //cout << "index1: " << index1 << " index2: " << index2;
    //initialize the bounds
    float Lbound = 0.0, Hbound = 0.0;
    //get bound
    check_condition(index1, index2, Lbound, Hbound);
    //cout << "Lbound: " << Lbound << "Hbound: " << Hbound;
    alpha2_new = update_alpha(index1, index2);
    //cout << "alpha2_new" << alpha2_new << endl;
    adjust_alpha(Lbound, Hbound);;
    //cout << "alpha2_new" << alpha2_new;
    alpha1_new = alpha1_old + labels[index1] * labels[index2] * (alpha2_old - alpha2_new);

    update_bias(index1, index2);

    alpha[index1] = alpha1_new;
    alpha[index2] = alpha2_new;
}

void SVM::check_condition(int index1, int index2,float &Lbound,float &Hbound) {
    if (labels[index1] != labels[index2]) {
        Lbound = max((float)0.0, alpha[index2] - alpha[index1]);
        Hbound = min(C, C + alpha[index2] - alpha[index1]);
    }
    else {
        Lbound = max((float)0.0, alpha[index2] + alpha[index1] - C);
        Hbound = min(C, alpha[index2] + alpha[index1]);
    }
}

float SVM::update_alpha(int index1, int index2) {
    float result;
    
    float sum = Kernel_function(inputs[index1], inputs[index1]) + Kernel_function(inputs[index2], inputs[index2]) + 2 * Kernel_function(inputs[index1], inputs[index2]);
    result = alpha2_old + (labels[index2] * (errors[index1] - errors[index2])) / sum;


    return result;
}

void SVM::adjust_alpha(float Lbound, float Hbound) {
    if (alpha2_new > Hbound) {
        alpha2_new = Hbound;
    }
    else if (alpha2_new < Lbound) {
        alpha2_new = Lbound;
    }
}

void SVM::update_bias(int index1, int index2) {
    float b1, b2;
    b1 = -errors[index1] - labels[index1] * Kernel_function(inputs[index1], inputs[index1]) * (alpha1_new - alpha1_old);
    b1 -= labels[index2] * Kernel_function(inputs[index2], inputs[index1]) * (alpha2_new - alpha2_old);
    b1 += bias;

    b2 = -errors[index2] - labels[index1] * Kernel_function(inputs[index1], inputs[index2]) * (alpha1_new - alpha1_old);
    b2 -= labels[index2] * Kernel_function(inputs[index2], inputs[index2]) * (alpha2_new - alpha2_old);
    b2 += bias;

    if (alpha1_new < C && alpha1_new>0) {
        bias = b1;
    }
    else if (alpha2_new < C && alpha2_new>0) {
        bias = b2;
    }
    else {
        bias = (b1 + b2) / 2;
    }
}
//select i which doesn't satisfy KKT condition
int SVM::select_alpha1() {
    for (int i = 0; i < alpha.size(); i++) {
        if ((alpha[i] > 0) && (alpha[i] < C) && (labels[i] * outputs[i] != 1)) {
            return i;
        }
        else if ((alpha[i] == 0) && (labels[i] * outputs[i] < 1)) {
            return i;
        }
        else if ((alpha[i] == C) && (labels[i] * outputs[i] > 1)) {
            return i;
        }
        else {
            //what if it satisfy kkt?
            return 0;
        }
    }
    return 0;
}

int SVM::select_alpha2(int index1) {
    float max = 0.0;
    int index;
    bool exist = false;
    for (int i = 0; i < updatelist.size(); i++) {
        if ((abs(errors[index1] - errors[updatelist[i]])>max)&&(index1!=updatelist[i])) {
            index = updatelist[i];
            max = abs(errors[index1] - errors[updatelist[i]]);
            exist = true;
        }
    }
    //if we can't find any, return random??
    if (!exist) {
        int len = errors.size();
        return rand() % len;
    }
    else {
        return index;
    }
}


vector<float> SVM::v_product(vector<float> v1, vector<float> v2) {
    vector<float> sum;
    if (v1.size() != v2.size()) {
        cout << "Vector size differ...";
    }
    else {
        for (int i = 0; i < v1.size(); i++) {
            sum.push_back(v1[i] * v2[i]);
        }
    }
    return sum;
}

float SVM::Kernel_function(vector<float> x1, vector<float> x2) {
    vector<float> sum = v_product(x1, x2);
    float result = 0.0;
    for (int i = 0; i < sum.size(); i++) {
        result += sum[i];
    }
    return result;
}

void SVM::print_alpha() {
    for (int i = 0; i < alpha.size(); i++) {
        cout << alpha[i] << " ";
    }
    cout << "Bias: " << bias << endl;
}

void SVM::print_v(vector<float> v) {
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;
}

void SVM::get_weights() {
    //The length of weights is the same amount as features
    //Each weight are the sum of alpha_j*output_j*input_j
    for (int i = 0; i < inputs[0].size(); i++) {
        float sumi = 0.0;
        for (int j = 0; j < inputs.size(); j++) {
            sumi += inputs[j][i] * outputs[j] * alpha[j];
        }
        weights[i] = sumi;
    }
}

void SVM::predict(vector<vector<float>> vec) {
    //re-initialize the weights
    get_weights();
    cout << "Output labels: ";
    for (int i = 0; i<vec.size(); i++){
        float sum_w = 0.0;
        for (int j = 0; j < vec[0].size(); j++) {
            sum_w += weights[j] * vec[i][j];
        }
        sum_w += bias;
        sum_w = sum_w >= 1 ? 1 : -1;
        cout << sum_w << " ";
    }

}
