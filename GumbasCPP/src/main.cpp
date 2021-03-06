
#include "GumbasCPP.h"


int main() {
    cout << "Loading model.\n";
    torch::jit::script::Module module = loadModel();

    vector<torch::jit::IValue> inputs = createTestInput(5,1);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    return 0;

}