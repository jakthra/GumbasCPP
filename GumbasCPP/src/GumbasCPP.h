﻿// GumbasCPP.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <fstream>
using namespace std;


torch::jit::script::Module loadModel();
vector<torch::jit::IValue> createTestInput(int batchSize, int dim);
// TODO: Reference additional headers your program requires here.
