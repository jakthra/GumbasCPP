
import torch
import torch.nn as nn
import numpy as np


class ToyNN(nn.Module):
    def __init__(self):
        super(ToyNN, self).__init__()

        self.input_layer = nn.Linear(1,5)
        self.output_layer = nn.Linear(5,1)

    def _init_weights(self):
        # initialize weights
        torch.nn.init.uniform_(self.input_layer.weight)
        torch.nn.init.uniform_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_layer(x)
        y = self.output_layer(x)
        return y



def create_nn_model():

    # Create model
    model = ToyNN()

    # initialize weights
    model._init_weights()
    print(model.input_layer.weight)
    print(model.output_layer.weight)

    # return model
    return model



if __name__ == "__main__":

    # Set seed
    seed = 1
    torch.manual_seed(seed)

    
    # Create simple NN
    # 1 -> 5 -> 5 -> 1
    model = create_nn_model()

    # Create test values
    test_input = torch.empty(50,1)
    test_input = torch.nn.init.uniform_(test_input)
    
    # Save test values
    np.savetxt('test_input.txt',test_input.numpy())

    # Forward pass
    test_output = model(test_input)
    np.savetxt('test_output.txt',test_output.detach().numpy())

    # Save model
    torch.save(model.state_dict(), 'toy_model.pt')

    traced_script_module = torch.jit.trace(model, test_input)
    traced_script_module.save("traced_toy_model.pt")
