// stdafx.cpp : source file that includes just the standard includes
// OpenNN.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

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

#include "../../../source/data_set/data_set.h"
#include "../../../source/data_set/instances_information.h"
#include "../../../source/data_set/variables_information.h"

// Neural Network includes

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

// Performance functional includes

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

// Training strategy includes

#include "../../../source/training_strategy/conjugate_gradient.h"
#include "../../../source/training_strategy/evolutionary_algorithm.h"
#include "../../../source/training_strategy/gradient_descent.h"
#include "../../../source/training_strategy/levenberg_marquardt_algorithm.h"
#include "../../../source/training_strategy/newton_method.h"
#include "../../../source/training_strategy/quasi_newton_method.h"
#include "../../../source/training_strategy/random_search.h"
#include "../../../source/training_strategy/training_algorithm.h"
#include "../../../source/training_strategy/training_rate_algorithm.h"


// TODO: reference any additional headers you need in STDAFX.H
// and not in this file
