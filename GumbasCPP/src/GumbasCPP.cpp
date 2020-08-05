// GumbasCPP.cpp : Defines the entry point for the application.
//

#include "GumbasCPP.h"


int loadModel() {

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load({ "models/traced_toy_model.pt" });
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	std::cout << "ok yes\n";

	//torch::Tensor tensor = torch::rand({2, 3});
	//std::cout << tensor << std::endl;
}