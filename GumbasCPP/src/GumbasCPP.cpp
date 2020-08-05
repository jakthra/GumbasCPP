// GumbasCPP.cpp : Defines the entry point for the application.
//

#include "GumbasCPP.h"


torch::jit::script::Module loadModel() {

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load({ "./models/traced_toy_model.pt" });
		
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}
	
	std::cout << "Loaded pytorch model.\n";
	return module;

	//torch::Tensor tensor = torch::rand({2, 3});
	//std::cout << tensor << std::endl;
}


vector<torch::jit::IValue> createTestInput(int batchSize, int dim) {
	//TODO: Load test inputs from file
	std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({batchSize, dim}));
	return inputs;
}



// TODO: load test outputs from file
// TOTO: Create test method for testing forward pass of model