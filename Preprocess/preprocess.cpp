//
//  preprocess.cpp
//  support vector machine
//
//  Created by Allen Chien on 5/24/23.
//

#include "preprocess.hpp"

preprocess::preprocess(){
    
}

preprocess::preprocess(string filename){
    ifstream file(filename);
    bool lab = true;
    
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        string value;
        vector<double> row;
        if(lab){
            while(getline(iss, value, ',')) {
                label.push_back(value);
            }
            lab = false;
            continue;
        }
        while (getline(iss, value, ',')) {
            try{
                row.push_back(stod(value));
                
            }catch (const std::exception& e) {
                cout << "Possible data loss: "<<value << e.what() << endl;
                double v = value =="M" ? 1.0:-1.0 ;
                row.push_back(v);
            }
        }
        raw_data.push_back(row);

    }
}

void preprocess::print_label(){
    for(int i=0;i<label.size();i++){
        printf("%s\n",label[i].c_str());
    }
}

void preprocess::show_data(){
    for(int i=0;i<label.size();i++){
        printf("%s\t",label[i].c_str());
    }
    for(int i=0;i<raw_data.size();i++){
        for(int j=0;j<raw_data[i].size();j++){
            printf("%f\t",raw_data[i][j]);
        }
        printf("\n");
    }
}

//new data = (old data - mean)/std deviation
void preprocess::stdscaler(int index){
    calculate_m_std(index);
    for(int i=0;i<raw_data.size();i++){
        raw_data[i][index] = (raw_data[i][index] - mean)/std;
    }
}

void preprocess::calculate_m_std(int index){
    double total = 0.0;
    for(int i=0;i<raw_data.size();i++){
        total += raw_data[i][index];
    }
    mean = total/raw_data.size();
    
    double sum = 0.0;
    for(int i=0;i<raw_data.size();i++){
        sum += (raw_data[i][index] - mean)*(raw_data[i][index] - mean);
    }
    std = sqrt(sum/raw_data.size());

}

vector<vector<double>> preprocess::read_column(int i1,int i2){
    vector<vector<double>> data;
    for(int i=0;i<raw_data.size();i++){
        data.push_back(vector<double>{raw_data[i][i1],raw_data[i][i2]});
    }
    return data;
}

vector<double> preprocess::read_column(int i1){
    vector<double> data;
    for(int i=0;i<raw_data.size();i++){
        data.push_back(raw_data[i][i1]);
    }
    return data;
}


vector<string> preprocess::read_label(int i1,int i2){
    return vector<string>{label[i1],label[i2]};
}

