#include <iostream>
#include <cstdlib>
#include "HebbianAutoencoder.h"

using namespace std;

int main()  {

    HebbianAutoencoder model;
    float error = 100.0;
    int epoch = 1;
    while(error > 0.5) {
        cout << "Epoch: " << epoch << endl;
        error = model.train();
    }
    return 0;
}
