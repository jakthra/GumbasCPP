﻿// GumbasCPP.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>


torch::jit::script::Module loadModel();
// TODO: Reference additional headers your program requires here.
