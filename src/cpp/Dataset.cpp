#include "Dataset.h"

MnistDataset::MnistDataset() {
    examples = new float * [60000];
    for(int i = 0; i < 60000; i++) {
        examples[i] = new float [784];
    }
    classifications = new float [60000];
}

MnistDataset::~MnistDataset() {

}

void MnistDataset::load(char * filename)  {
    ifstream mnist;
    mnist.open(filename);
    for(int i = 0; i < 60000; i++) {
        for(int j = 0; j < 784; j++)
            mnist >> examples[i][j];
        mnist >> classifications[i];
    }
}

float * MnistDataset::get_example(int index)  {
    return examples[index];
}

float MnistDataset::get_classifications(int index)  {
    return classifications[index];
}

