#include "bolzman.h"

bool RBM::RBM_train()
{
	if(!ReadyToTrane)
		return false;
	//-----Compute first hiden probs and states
	computeProbs(0);
	computeHidStates();
	//-----
	//-----start epoches
	for(int k=0;k<Epochs;k++){
		for(int i=0;i<NumHidSt;i++)
			PosHidStates[i]=HidStates[i];
		for(int i=0;i<NumVisSt;i++)
			PosVisStates[i]=VisStates[i];

		//start gibbs sempling
		for(int j=0;j<15;j++){
			computeProbs(1);
			computeVisStates();
			computeProbs(0);
			computeHidStates();
		}

		updateWeights();
		updateVisiableBiases();
		computeError();

		if(Error<Eps)
			break;
	}
	//-----
	return true;
}

void	RBM::initBias()
{
	switch (biasInitTipe) {
		case 0:
			for(int i=0;i<NumHidSt;i++)
				VisBiases[i]=0;
			break;
		default:
			for(int i=0;i<NumHidSt;i++)
				VisBiases[i]=rand()/(RAND_MAX/2)-1;
			break;
	}
}

void	RBM::initWeights()
{
	for(int i=0;i<NumVisSt;i++){
		for(int j=0;j<NumHidSt;j++)
			Weights[i][j]=rand()/(RAND_MAX/2)-1;
	}
}

void	RBM::initVisiableStates()
{
	for(int i=0;i<NumVisSt;i++)
		VisStates[i]=Data[i];
}

void	RBM::computeProbs(unsigned short flag)
{
	switch(flag){
		case 0:
			for(int i=0;i<NumHidSt;i++){
				HidProbs[i]=actFunc(hidProbSum(i));
			}
			break;
		case 1:
			for(int i=0;i<NumVisSt;i++){
				VisProbs[i]=actFunc(visProbSum(i));
			}
			break;
		default:
			break;
	}
}

double	RBM::actFunc(double value)
{
	return 1/(1+exp(value));
}

double	RBM::hidProbSum(unsigned int hidMemb)
{
	double sum=0;
	for(int i=0;i<NumVisSt;i++)
		sum+=VisStates[i]*Weights[i][hidMemb];
	return sum+=VisBiases[hidMemb];
}

double	RBM::visProbSum(unsigned int visMemb)
{
	double sum=0;
	for(int i=0;i<NumHidSt;i++)
		sum+=HidStates[i]*Weights[visMemb][i];
	return sum;
}

void RBM::computeHidStates()
{
	double randVal;
	for(int i=0;i<NumHidSt;i++){
		randVal=rand()/RAND_MAX;
		HidStates[i]=(randVal>HidProbs[i])?0:1;
	}
}

void RBM::computeVisStates()
{
	double randVal;
	for(int i=0;i<NumHidSt;i++){
		randVal=rand()/RAND_MAX;
		VisStates[i]=(randVal>VisProbs[i])?0:1;
	}
}

void RBM::updateWeights()
{
	for(int i=0;i<NumVisSt;i++){
		for(int j=0;j<NumHidSt;j++)
			Weights[i][j]+=LearningRate*(PosHidStates[j]*PosVisStates[i]-HidStates[j]*VisStates[i]);
	}
}

void RBM::updateVisiableBiases()
{
	for(int i=0;i<NumHidSt;i++)
		VisBiases[i]+=LearningRate*(PosVisStates[i]-VisStates[i]);
}

void RBM::computeError()
{
	for(int i=0;i<NumHidSt;i++){
		Error+=pow(PosHidProbs[i]-HidProbs[i],2);
	}
}

RBM::RBM(vectorDouble _data,
		 unsigned int _num_hidden,
		 unsigned int _num_visible,
		 double _learning_rate,
		 double _eps,
		 unsigned int _epochs )
{
	NumHidSt=_num_hidden;
	NumVisSt=_num_visible;
	LearningRate=_learning_rate;
	Data=_data;
	Eps=_eps;
	Epochs=_epochs;

	HidStates=new double[NumHidSt];
	VisStates=new double[NumVisSt];
	VisBiases=new double[NumHidSt];
	PosHidStates=new double[NumHidSt];
	PosVisStates=new double[NumVisSt];

	VisProbs=new double[NumVisSt];
	HidProbs=new double[NumHidSt];
	PosHidProbs=new double[NumHidSt];
	Weights=(double**)malloc(NumVisSt);
	for(int i=0;i<NumVisSt;i++){
		Weights[i]=new double[NumHidSt];
	}

	initBias();
	initWeights();
	initVisiableStates();
	ReadyToTrane=true;

}

RBM::~RBM()
{
	delete HidStates;
	delete VisStates;
	delete VisBiases;
	delete PosHidStates;
	delete PosVisStates;

	delete VisProbs;
	delete HidProbs;
	delete PosHidProbs;
	for(int i=0;i<NumVisSt;i++){
		delete Weights[i];
	}
	delete Weights;
	ReadyToTrane=false;
}
