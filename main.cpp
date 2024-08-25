#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <memory>
#include "../libtorch/include/torch/csrc/jit/serialization/import.h"

int main(int argc, const char *argv[])
{
    //! Load the model
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load("colision_detector.pt");
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    //! Ensure the model is in evaluation mode
    module.eval();
    module.to(at::kCPU);

    //! Disable gradient computation
    torch::NoGradGuard no_grad;

    auto start = std::chrono::high_resolution_clock::now();
    int NUM_ITERATIONS = 1000;
    for (int index = 0; index < NUM_ITERATIONS; index++)
    {
        torch::Tensor input_tensor = torch::randn({1, 3});
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        //! Forward pass
        torch::Tensor output = module.forward(inputs).toTensor();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "ok\n";
    std::cout << "Inference Time: " << duration.count() / 1000.0 << " microseconds" << std::endl;
}