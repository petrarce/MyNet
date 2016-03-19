#include <vector>
#include <ctime>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <iostream>
#include <sys/stat.h>

typedef char**		matrixBool;
typedef double**	matrixDouble;
typedef double*		vectorDouble;

class RBM
{
   private:
		//bool net_stat;					//??
		bool ReadyToTrane=false;			//training flag
		bool biasInitTipe;
		double LearningRate;

		double Error;
		double Eps=0.0001;

		//ofstream log_file;

		unsigned int NumHidSt;
		unsigned int NumVisSt;
		//unsigned int train_data_rows;
		//unsigned int train_data_cols;

		uint32_t CurrEpoch;
		uint32_t Epochs;

		vectorDouble Data;
		vectorDouble HidStates;
		vectorDouble VisStates;
		vectorDouble PosHidStates;
		vectorDouble PosVisStates;

		matrixDouble Weights;		//Weight[i][j] i-visiable member, j-hedden member
		vectorDouble VisBiases;

		//vectorDouble PosAssociations;
		//vectorDouble NegAssociations;

		vectorDouble VisProbs;
		vectorDouble HidProbs;
		vectorDouble PosHidProbs;

	public:

		/* Constructor Functions */
		RBM();
		RBM(vectorDouble _data, unsigned int _num_hidden, unsigned int _num_visible, double _learning_rate);
		~RBM();


		/* RBM Initialization Functions */
		void initBias();			//no biases for hidden members
		void initWeights();
		void initVisiableStates();			//int with data

		/* RBM Parameters Configuration Functions */

		/* RBM core Functions */
		void computeError();
		void updateWeights();
		void updateVisiableBiases();

		double actFunc(double value);
		void computeNegAssociations();
		void computePosAssociations();
		void computeProbs(unsigned short flag);	//visiable-1;hidden-0;
		void computeVisStates();
		void computeHidStates();
		bool RBM_train(unsigned int epochs);
	private :
		double hidProbSum(unsigned int hidMemb);
		double visProbSum(unsigned int visMemb);
};
