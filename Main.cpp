/*
 * Main.cpp
 *
 *  Created on: 25 Dec 2010
 *      Author: thomas
 */

#include <vector>
#include <limits>
#include <sys/time.h>
#include <random>
#include "BackPropagation-master/src/Backpropagation.h"
#include "bolzman/bolzman.h"

using namespace std;

int main()
{
	//-----Create network pattern
	const int myNetArray[3] = {20, 2, 1};
    vector<int> myNet(myNetArray, myNetArray + sizeof(myNetArray) / sizeof(int));
	//-----
	//-----Create input-output paterns pattern

	const int patternSize = 6;

	const double myTestOne[patternSize] = {1, 1,0,1,0,0};
    double oneOut = 0;
	const double myTestTwo[patternSize] = {1, 0,1,1,0,0};
    double twoOut = 1;
	const double myTestThree[patternSize] = {0, 1,1,1,1,1};
	double threeOut = 1;
	const double myTestFour[patternSize] = {0, 0,0,1,1,0};
    double fourOut = 0;

    Pattern firstPattern(patternSize, (double *)&myTestOne, &oneOut);
    Pattern secondPattern(patternSize, (double *)&myTestTwo, &twoOut);
    Pattern thirdPattern(patternSize, (double *)&myTestThree, &threeOut);
    Pattern fourthPattern(patternSize, (double *)&myTestFour, &fourOut);
	//-----Create network with using network pattern and size of inpet data
	Backpropagation * myNeuralNetwork = new Backpropagation(myNet, patternSize);
	//-----

	//-----Init weights using RBM
//	double* Data;//=(double*)malloc(myNeuralNetwork->numLayers);
//	RBM myRbms;
//	for(int i=0;i<myNeuralNetwork->numLayers;i++){
//		//---Create Dataset
//		Data=new double[myNeuralNetwork->layers[i].numInputs];
//		for(int j=0;j<myNeuralNetwork->layers[i].numInputs;j++){
//			if(!j){
//				Data[j]=1;
//				continue;
//			}
//			Data[j]=(Data[j-1]==1)?0:1;
//		}
//		//---
//		myRbms.initRBM(Data,
//					   myNeuralNetwork->layers[i].numNeurons,
//					   myNeuralNetwork->layers[i].numInputs,
//					   0.1,
//					   0.01,
//					   3000);
//		myRbms.RBM_train();
//		for(unsigned int j=0;j<myRbms.NumHidSt;j++)
//			for(unsigned int k=0;k<myRbms.NumVisSt;k++)
//				myNeuralNetwork->layers[i].neurons[j].weights[k]=myRbms.Weights[k][j];
//		myRbms.~RBM();
//		delete[] Data;
//	}
	//-----

    double errors[4];
    double currentError = numeric_limits<double>::max();
    unsigned long long int iteration = 0;

    timeval timer;

    gettimeofday(&timer, 0);

    double start = (timer.tv_sec * 1000000) + timer.tv_usec;
	//-----Training of neural network

    while(currentError > 0.0001)
    {
        errors[0] = myNeuralNetwork->train(firstPattern);
        errors[1] = myNeuralNetwork->train(secondPattern);
        errors[2] = myNeuralNetwork->train(thirdPattern);
        errors[3] = myNeuralNetwork->train(fourthPattern);

        double largest = errors[0];

        for(int i = 0; i < 4; i++)
        {
            if(errors[i] > largest)
            {
                largest = errors[i];
            }
        }

        if(largest < currentError)
        {
            currentError = largest;
        }

        iteration += 4;
    }
	//-----

    gettimeofday(&timer, 0);

    double end = (timer.tv_sec * 1000000) + timer.tv_usec;


    cout << "Error: " << currentError << ", iterations: " << iteration << endl;
    cout << "Iterations per second: " << iteration / ((end - start) / 1000000) << endl;

    cout << "Testing: " << endl;
    cout << "Input | Output" << endl;
    myNeuralNetwork->feedForward((double *)&myTestOne);
    cout << myTestOne[0] << " " << myTestOne[1] << " | " << myNeuralNetwork->getOutput()[0] << endl;
    myNeuralNetwork->feedForward((double *)&myTestTwo);
    cout << myTestTwo[0] << " " << myTestTwo[1] << " | " << myNeuralNetwork->getOutput()[0] << endl;
    myNeuralNetwork->feedForward((double *)&myTestThree);
    cout << myTestThree[0] << " " << myTestThree[1] << " | " << myNeuralNetwork->getOutput()[0] << endl;
    myNeuralNetwork->feedForward((double *)&myTestFour);
    cout << myTestFour[0] << " " << myTestFour[1] << " | " << myNeuralNetwork->getOutput()[0] << endl;

    delete myNeuralNetwork;
    return 0;
}
