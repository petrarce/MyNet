// OpenNN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


// System includes

#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <ostream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>


// Utilities includes

#include "../../../source/Utilities/Vector.h"
#include "../../../source/Utilities/Matrix.h"
#include "../../../source/utilities/linear_algebraic_equations.h"
#include "../../../source/utilities/numerical_differentiation.h"
#include "../../../source/utilities/numerical_integration.h"

// Data Set

#include "../../../source/data_set/data_set.h"
#include "../../../source/data_set/instances_information.h"
#include "../../../source/data_set/variables_information.h"

// Mathematical Model

#include "../../../source/mathematical_model/mathematical_model.h"
#include "../../../source/mathematical_model/ordinary_differential_equations.h"
#include "../../../source/mathematical_model/plug_in.h"

// Model Selection

#include "../../../source/model_selection/model_selection.h"

// Neural Network

#include "../../../source/neural_network/bounding_layer.h"
#include "../../../source/neural_network/conditions_layer.h"
#include "../../../source/neural_network/independent_parameters.h"
#include "../../../source/neural_network/inputs_outputs_information.h"
#include "../../../source/neural_network/multilayer_perceptron.h"
#include "../../../source/neural_network/neural_network.h"
#include "../../../source/neural_network/perceptron.h"
#include "../../../source/neural_network/perceptron_layer.h"
#include "../../../source/neural_network/probabilistic_layer.h"
#include "../../../source/neural_network/scaling_layer.h"
#include "../../../source/neural_network/unscaling_layer.h"

// Perfomance Functional

#include "../../../source/performance_functional/cross_entropy_error.h"
#include "../../../source/performance_functional/final_solutions_error.h"
#include "../../../source/performance_functional/independent_parameters_error.h"
#include "../../../source/performance_functional/inverse_sum_squared_error.h"
#include "../../../source/performance_functional/mean_squared_error.h"
#include "../../../source/performance_functional/minkowski_error.h"
#include "../../../source/performance_functional/neural_parameters_norm.h"
#include "../../../source/performance_functional/normalized_squared_error.h"
#include "../../../source/performance_functional/outputs_integrals.h"
#include "../../../source/performance_functional/performance_functional.h"
#include "../../../source/performance_functional/performance_term.h"
#include "../../../source/performance_functional/root_mean_squared_error.h"
#include "../../../source/performance_functional/solutions_error.h"
#include "../../../source/performance_functional/sum_squared_error.h"

// Testing Analysis

#include "../../../source/testing_analysis/function_regression_testing.h"
#include "../../../source/testing_analysis/inverse_problem_testing.h"
#include "../../../source/testing_analysis/pattern_recognition_testing.h"
#include "../../../source/testing_analysis/testing_analysis.h"
#include "../../../source/testing_analysis/time_series_prediction_testing.h"

// Training Strategy

#include "../../../source/training_strategy/conjugate_gradient.h"
#include "../../../source/training_strategy/evolutionary_algorithm.h"
#include "../../../source/training_strategy/gradient_descent.h"
#include "../../../source/training_strategy/levenberg_marquardt_algorithm.h"
#include "../../../source/training_strategy/newton_method.h"
#include "../../../source/training_strategy/quasi_newton_method.h"
#include "../../../source/training_strategy/random_search.h"
#include "../../../source/training_strategy/training_algorithm.h"
#include "../../../source/training_strategy/training_rate_algorithm.h"
#include "../../../source/training_strategy/training_strategy.h"



#define Max_ne 1000
#define Max_ns 900
#define Max_nval 1000

using namespace Flood;
using namespace std;

int main(void)
{
   int ne, ns, mt, dspd, epc, mihn, mahn, t_ann;
   double eg, gn, pertr, perax, pervl;
   double aux, ermig,minermig;
   int i,j,k,nval, sel;
   size_t length;
   errno_t er0, er1, er2;
   std::vector<double> inpt;
   Vector<double> output(Max_ns, 0.0);
   Vector<double> val(Max_ns, 0.0);
   Vector<double> minerr(Max_ns, 0.0);
   string nmlp;
   string nom;
   string nmaster;
   MultilayerPerceptron mlp;
   char instch[100];
   char ndataset[50], nomaster[50];
   FILE *enf, *srt, *ferror;
	

   std::cout << std::endl << "OpenNN Neural Network." << std::endl;
   std::cout << "Generic Metamodel Application." << std::endl;	
   srand((unsigned)time(NULL));

   // PASO 0 Definición del modelo

   std::cout << std::endl << "####################################################################" << std::endl;
   std::cout << "#" << std::endl;
   std::cout << "#       Reading the configuration for the new DSS ANN" << std::endl;
   std::cout << "#       Reading file: config_ANN.dat" << std::endl;
   string inputFile = "config_ANN.dat";
   ifstream inF(inputFile.c_str());
   if (!inF){
	   std::cout << "!!!!! Couldn't open the config file." << std::endl;
	   cout << "!!!!! Please check config_ANN.dat exists and is readable" << endl;
	   exit(1);
   }
   inF >> nmlp;
   std::cout << std::endl << "# MLP file: " << nmlp << "\n";
   inF.ignore(255, '\n');
   inF >> ne;
   inF.ignore(255, '\n');
   inF >> ns;
   std::cout << "# Input Variables: " << ne << std::endl << "# Output Variables: " << ns << std::endl;
   inF.ignore(255, '\n');
   inF >> mihn;
   inF.ignore(255, '\n');
   inF >> mahn;
   std::cout << "# Min hidden neurons: " << mihn << std::endl << "# Max hidden neurons: " << mahn << std::endl;
   inF.ignore(255, '\n');
   inF >> t_ann;
   std::cout << "# Type of ANN: " << t_ann;
   if(t_ann==0){
	   std::cout << " Function_regression" << std::endl;
   else if(t_ann==1){
	   std::cout << " Pattern_recognition" << std::endl;
   else if(t_ann==2){
	   std::cout << " Time_series_prediction" << std::endl;
   else if(t_ann==3){
	   std::cout << " Inverse_problem" << std::endl;
   }
   inF.ignore(255, '\n');
   inF >> nom;
   length=nom.copy(ndataset,49,0);
   ndataset[length]='\0';
   std::cout << "# Name of dataset: " << nom << std::endl;
   inF.ignore(255, '\n');
   inF >> eg;
   inF.ignore(255, '\n');
   inF >> gn;
   inF.ignore(255, '\n');
   inF >> mt;
   inF.ignore(255, '\n');
   inF >> epc;
   inF.ignore(255, '\n');
   inF >> dspd;
   std::cout << "# EvaluationGoal: " << eg << std::endl << "# GradientNormGoal: " << gn << std::endl;
   std::cout << "# MaximumTime: " << mt << std::endl << "# MaximumNumberOfEpochs: " << epc << std::endl;
   std::cout << "# displayperiod: " << dspd << std::endl;
   inF.ignore(255, '\n');
   inF >> pertr;
   inF >> perax;
   inF >> pervl;
   inF.ignore(255, '\n');
   std::cout << "# Instance Training: " << pertr << "%" << std::endl;
   std::cout << "# Instance Generalization: " << perax << "%" << std::endl;
   std::cout << "# Instance Validation: " << pervl << "%" << std::endl;
   inF >> nmaster;
   length=nmaster.copy(nomaster,49,0);
   nomaster[length]='\0';
   std::cout << "# Master File: " << nmaster << std::endl;
   inF.ignore(255, '\n');

   
   // PASO 1 CREAR EL METAMODELO

   Vector<double> erro(Max_ns, 0.0);
   sel=0;
   minermig=1000000.0;
   
   // Data set 

   DataSet data_set;
   data_set.load_data(ndataset);
   
   // Variables information
   VariablesInformation* variables_information_pointer = data_set.get_variables_information_pointer();

   for(j=0;j<ne;j++){
	   inF >> nomvar;
	   std::cout << "# Name of Variable " << j << ": " << nomvar << std::endl;
	   inF >> unitvar;
	   std::cout << "# Units of Variable " << j << ": " << unitvar << std::endl;
	   inF.ignore(255, '\n');
	   variables_information_pointer->set_name(j, nomvar);
	   if(unitvar!="0"){
		   variables_information_pointer->set_units(j, unitvar);
	   }
   }
   inF.close();

   const Vector< Vector<std::string> > inputs_targets_information = variables_information_pointer->arrange_inputs_targets_information();

   // Instances information

   InstancesInformation* instances_information_pointer = data_set.get_instances_information_pointer();

   instances_information_pointer->split_random_indices(pertr, perax, pervl);

   const Vector< Vector<double> > inputs_targets_statistics = data_set.scale_inputs_minimum_maximum();

   //data_set.print_data();

   for (i=mihn; i<mahn+1; i++){
	  // Neural network 
	  
      NeuralNetwork neural_network(ne, i, ns);

      neural_network.set_inputs_outputs_information(inputs_targets_information); 
      neural_network.set_inputs_outputs_statistics(inputs_targets_statistics); 

	  neural_network.set_scaling_unscaling_layers_flag(false);

      // Performance functional

      PerformanceFunctional performance_functional(&neural_network, &data_set);

      // Training strategy

      TrainingStrategy training_strategy(&performance_functional);

      TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

	  if(t_ann==0){
		  neural_network.set_inputs_scaling_outputs_unscaling_methods("MeanStandardDeviation");
	  else if(t_ann==1){
		  neural_network.set_scaling_layer_flag(true);
	  else if(t_ann==2){
		  
	  else if(t_ann==3){
		  
	  }
	  
	  // Testing analysis

	  TestingAnalysis testing_analysis(&neural_network, &data_set);

	  if(t_ann==0){
		  testing_analysis.construct_function_regression_testing();
		  FunctionRegressionTesting* function_regression_testing_pointer = testing_analysis.get_function_regression_testing_pointer();
		  FunctionRegressionTesting::LinearRegressionAnalysisResults linear_regression_analysis_results = testing_analysis.get_function_regression_testing_pointer()->perform_linear_regression_analysis();
		  std::cout << "Linear regression parameters:\n" 
		        << "Intercept: " << linear_regression_analysis_results.linear_regression_parameters[0][0] << "\n"
				<< "Slope: " << linear_regression_analysis_results.linear_regression_parameters[0][1] << std::endl;

	  else if(t_ann==1){
		  testing_analysis.construct_pattern_recognition_testing();
		  PatternRecognitionTesting* pattern_recognition_testing_pointer = testing_analysis.get_pattern_recognition_testing_pointer();
	  else if(t_ann==2){
		  testing_analysis.construct_time_series_prediction_testing();
		  TimeSeriesPredictionTesting* time_series_prediction_testing_pointer = testing_analysis.get_time_series_prediction_testing_pointer();
	  else if(t_ann==3){
		  testing_analysis.construct_inverse_problem_testing();
		  InverseProblemTesting* inverse_problem_testing_pointer = testing_analysis.get_inverse_problem_testing_pointer();
	  }
      
      // Save results 

	  data_set.save(nmlp+"_"+"data_set.xml");

      neural_network.save(nmlp+"_"+"neural_network.xml");
      neural_network.save_expression(nmlp+"_"+"expression.txt");

	  performance_functional.save(nmlp+"_"+"performance_functional.xml");

      training_strategy.save(nmlp+"_"+"training_strategy.xml");
      training_strategy_results.save(nmlp+"_"+"training_strategy_results.dat");

      if(t_ann==0){
		  linear_regression_analysis_results.save(nmlp+"_"+"linear_regression_analysis_results.dat");
	  else if(t_ann==1){
		  pattern_recognition_testing_pointer->save_confusion(nmlp+"_"+"confusion.dat");
		  pattern_recognition_testing_pointer->save_binary_classification_test(nmlp+"_"+"binary_classification_test.dat");
	  else if(t_ann==2){
		  
	  else if(t_ann==3){
		  
	  }
	     


/*   string instr1 = "MlP";
	   string instr2 = ".dat";
	   string numsel = static_cast<ostringstream*>( &(ostringstream() << i) )->str();
	   string instruc =nmlp+instr1+numsel+instr2;
	   length=instruc.copy(instch,49,0);
	   instch[length]='\0';
	   multilayerPerceptron.save(instch);*/
	}
/*
   // PASO 2: Validación del modelo

	cout << endl << "Validation process starts" << endl;
	er2=fopen_s(&ferror, "training_errors.dat", "wt");
		   
	if(er2){	std::cout << "Training_errors.dat not opened " << std::endl;
	}else if(!er2){
		for(j=mihn;j<mahn+1;j++){
			string instr1 = "MlP";
			string instr2 = ".dat";
			string numsel = static_cast<ostringstream*>( &(ostringstream() << j) )->str();
			string instruc =nmlp+instr1+numsel+instr2;
			length=instruc.copy(instch,49,0);
			instch[length]='\0';
			mlp.load(instch);
			
			cout << endl << "MLP Loaded" << endl;
			InputCalculDataSet icds;
			er0=fopen_s(&enf, nomaster,"rt");
			std::cout << "Opening Master file: " << nmaster << std::endl;
			if (!enf){	std::cout << "Couldn't open the Master file." << std::endl;
			}else if(enf){
				string instr1 = "VAL";
				string instr2 = ".out";
				string numsel = static_cast<ostringstream*>( &(ostringstream() << j) )->str();
				string instruc =nmlp+instr1+numsel+instr2;
				length=instruc.copy(instch,49,0);
				instch[length]='\0';
				er1=fopen_s(&srt, instch, "wt");
				
				fscanf(enf, "%d", &nval);
				Vector<double> inpv(ne,0.0);
				ermig=0.0;
			    InputCalculDataSet::InputCalculDataSet(ne,ns);
				for(i=0;i<nval;i++){
					for(k=0;k<ne;k++){
						fscanf(enf, "%f", &aux);
						inpt.push_back(aux);
						inpv[k]=inpt[k];
					}
					std::cout << std::endl << "Validation Input Data ok " << std::endl;
					output = mlp.getOutput(inpv);
					for(k=0;k<ns;k++){
						fscanf(enf, "%f", &aux);
						erro[k] = erro[k]+fabs(output[k]-aux);
					}
					std::cout << "Validation Output Data ok\n " << std::endl;
				}
				fprintf(ferror,"%d ",j);
				for(k=0;k<ns;k++){
					erro[k]=erro[k]/nval;
					fprintf(ferror,"%f ",erro[k]);
					ermig=+erro[k];
				}
				ermig=ermig/ns;
				fprintf(ferror,"--> %f\n",ermig);
				if(ermig<=minermig){
					sel=j;
					minermig=ermig;
				}
				
			}else perror("Error reading Master file");
			fclose(srt);
		}
	}
	string instr0 = "copy ";
	string instr1 = "MlP";
	string instr2 = ".dat ";
	string instr3 = ".dat";
	string numsel = static_cast<ostringstream*>( &(ostringstream() << sel) )->str();
	string instruc =instr0+nmlp+instr1+numsel+instr2+nmlp+instr3;
	length=instruc.copy(instch,49,0);
	instch[length]='\0';
	system(instch);
	std::cout << std::endl << "Selecting " << sel << " hidden neurons" << std::endl;
	std::cout << std::endl << "Neural Network Generated" << std::endl;
	fclose(enf);
	fclose(ferror);
	*/
	return(0);
}


