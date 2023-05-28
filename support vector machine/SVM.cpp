#include "SVM.hpp"

double SVM::bias = 0.0;
double SVM::thresh = 5.0;

vector<double> SVM::weights = { 0.0,0.0 };
vector<vector<double>> SVM::inputs = { {} };
vector<double> SVM::labels = {};


SVM::SVM(vector<vector<double>>inputs, vector<double>labels, double C, double tolerence) {
    this->inputs = inputs;
    this->labels = labels;
    this->C = C;
    this->tol = tolerence;
    int len = inputs.size();
    //initialize the vectors with the same size as input, including the alpha, output, error vector
    init_alpha_b(len);
}

void SVM::init_alpha_b(int len) {
    this->bias = 0.0;
    alpha = vector<double>(len, 0.0);
    outputs = vector<double>(len, 0.0);
    errors = vector<double>(len, 0.0);
    weights = vector<double>(inputs[0].size(), 0.0);
}

//sumof(i=1,n)(alpha_i*y_i*K_i_x) = w T x +b
void SVM::get_output() {
    for (int i = 0; i < outputs.size(); i++) {
        //calculate the output for each input
        double sum = 0.0;
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
    for (int i = 0; i < epoch; i++) {
        get_output();
        get_error();
        update();
    }
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
    double Lbound = 0.0, Hbound = 0.0;
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

void SVM::check_condition(int index1, int index2,double &Lbound,double &Hbound) {
    if (labels[index1] != labels[index2]) {
        Lbound = max((double)0.0, alpha[index2] - alpha[index1]);
        Hbound = min(C, C + alpha[index2] - alpha[index1]);
    }
    else {
        Lbound = max((double)0.0, alpha[index2] + alpha[index1] - C);
        Hbound = min(C, alpha[index2] + alpha[index1]);
    }
}

double SVM::update_alpha(int index1, int index2) {
    double result;
    
    double sum = Kernel_function(inputs[index1], inputs[index1]) + Kernel_function(inputs[index2], inputs[index2]) + 2 * Kernel_function(inputs[index1], inputs[index2]);
    result = alpha2_old + (labels[index2] * (errors[index1] - errors[index2])) / sum;


    return result;
}

void SVM::adjust_alpha(double Lbound, double Hbound) {
    if (alpha2_new > Hbound) {
        alpha2_new = Hbound;
    }
    else if (alpha2_new < Lbound) {
        alpha2_new = Lbound;
    }
}

void SVM::update_bias(int index1, int index2) {
    double b1, b2;
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
    double max = 0.0;
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


vector<double> SVM::v_product(vector<double> v1, vector<double> v2) {
    vector<double> sum;
    try {
        if (v1.size() != v2.size()) {
            throw v1.size();
        }
        for (int i = 0; i < v1.size(); i++) {
            sum.push_back(v1[i] * v2[i]);
        }
        return sum;
    }
    catch (int size) {
        cout << "Caution: vector size"<<size<< "and "<<v2.size()<<"different but used in multiplication.";
        return sum;
    }
}

double SVM::Kernel_function(vector<double> x1, vector<double> x2) {
    vector<double> sum = v_product(x1, x2);
    double result = 0.0;
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

void SVM::print_v(vector<double> v) {
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }
    cout << endl;
}

void SVM::get_weights() {
    //The length of weights is the same amount as features
    //Each weight are the sum of alpha_j*output_j*input_j
    for (int i = 0; i < inputs[0].size(); i++) {

        double sumi = 0.0;
        for (int j = 0; j < inputs.size(); j++) {
            sumi += inputs[j][i] * outputs[j] * alpha[j];
        }
        weights[i] = sumi;
    }
}

void SVM::predict(vector<vector<double>> vec) {
    //re-initialize the weights
    get_weights();
    cout << "Output labels: ";
    for (int i = 0; i<vec.size(); i++){
        double sum_w = 0.0;

        for (int j = 0; j < vec[0].size(); j++) {
            sum_w += weights[j] * vec[i][j];
        }
        sum_w += bias;
        sum_w = sum_w >= 0 ? 1 : -1;
        cout << sum_w << " ";
    }
}

void SVM::plot(int argc, char** argv) {
    glutInit(&argc, argv);                // Initialize GLUT
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);     // Set display mode (single buffer)
    glutInitWindowSize(500, 500); // Set initial window size
    glutInitWindowPosition(10, 10);

    string title = "Linear SVM weights:"+to_string(weights[0])+","+to_string(weights[1])+" bias:"+to_string(bias);
    glutCreateWindow("svm");    // Create the window
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-thresh, thresh, -thresh, thresh);

    glutDisplayFunc(display);            // Set the display callback function

    glutMainLoop();                       // Enter the event loop
}

void SVM::display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(2.0);

    glLineWidth(1.0); // Set line width to make it more visible

    // Draw x-axis
    glColor3f(0.0, 0.0, 0.0); // Set color to black
    glBegin(GL_LINES);
    glVertex2d(-thresh, 0.0);
    glVertex2d(thresh, 0.0);
    glEnd();

    // Draw y-axis
    glColor3f(0.0, 0.0, 0.0); // Set color to black
    glBegin(GL_LINES);
    glVertex2d(0.0, -thresh);
    glVertex2d(0.0, thresh);
    glEnd();

    glLineWidth(1.0); // Reset line width

    glColor3f(1.0, 0.0, 0.0);

    glBegin(GL_LINES);
    cout<<weights[0]<<" "<<weights[1]<<" "<<bias<<endl;
    //wTx+b=0
    //(w1,w2)T(x1,x2)+b=0
    //w1*x1+w2*x2+b=0
    //x2=(w1*x1+b)/-w2
    double y1 = ((weights[0] * -100.0) + bias) / (-weights[1]+0.001);
    glVertex2d(-thresh, y1);
    double y2 = ((weights[0] * 100.0) + bias) / (-weights[1]+0.0001);
    glVertex2d(thresh, y2);
    cout<<y1<<" "<<y2<<endl;
    glEnd();

    display_points();

    glFlush();
}

void SVM::display_points() {
    glPointSize(5.0f); // Set dot size

    glBegin(GL_POINTS);
    glBegin(GL_POINTS);
    //glVertex2d(0.0, 0.0f); // Dot at (0, 0)
    for (int i = 0; i < inputs.size(); i++) {
        if(labels[i]==1)    glColor3f(0.0, 1.0, 0.0);
        else    glColor3f(0.0, 0.0, 1.0);

        glVertex2d(inputs[i][0], inputs[i][1]);
    }
    glEnd();
}
