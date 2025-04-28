#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h" // Assuming nn.h includes necessary headers

int main(void) {
    nn_log(LOG_INFO, "--- Minimal NN Tester: Complex Square Root Fit ---");

    // --- Configuration ---
    size_t n_inputs = 1;  // **MOVED** to variable
    size_t n_outputs = 2; // **MOVED** to variable - real and imaginary parts
    size_t epochs = 150000;
    size_t train_batch_size = 10;
    pfloat learning_rate = 0.01L;
    int random_seed = 987;

    Model *model = NULL;
    Optimizer *opt = NULL;
    Params params;

    // --- Complex Square Root Dataset ---
    pfloat inputs[] = {
        -4.0L,
        -1.0L,
         0.0L,
         1.0L,
         4.0L,
         0.25L,
        -0.25L,
         2.0L,
        -2.0L,
         0.5L
    };
    pfloat targets[] = {
        0.0L, 2.0L,     // sqrt(-4) = 0 + 2i
        0.0L, 1.0L,     // sqrt(-1) = 0 + 1i
        0.0L, 0.0L,     // sqrt(0) = 0 + 0i
        1.0L, 0.0L,     // sqrt(1) = 1 + 0i
        2.0L, 0.0L,     // sqrt(4) = 2 + 0i
        0.5L, 0.0L,     // sqrt(0.25) = 0.5 + 0i
        0.0L, 0.5L,     // sqrt(-0.25) = 0 + 0.5i
        sqrtl(2.0L), 0.0L,  // sqrt(2) = real positive
        0.0L, sqrtl(2.0L),  // sqrt(-2) = imaginary positive
        sqrtl(0.5L), 0.0L   // sqrt(0.5)
    };
    size_t num_samples = sizeof(inputs) / (n_inputs * sizeof(pfloat));
    nn_log(LOG_INFO, "Using Complex Square Root dataset (%zu samples)", num_samples);

    // --- Configure and Create Model ---
    opt = make_optimizer(learning_rate);
    if (!opt) {
        nn_log(LOG_ERROR, "Failed to create optimizer.");
        return 1;
    }

    // **UPDATED** Params struct initialization (removed n_inputs, n_outputs)
    params = (Params){
        // .n_inputs = n_inputs,  // No longer needed here
        // .n_outputs = n_outputs, // No longer needed here
        .loss = &nnMSE,         // still MSE over 2 outputs
        .optimizer = opt,
        .seed = random_seed,
        .log_frequency_epochs = 3000,
        .randomize_batches = false
    };

    // Model structure: 1 -> 12 -> 12 -> 2
    // **UPDATED** make_model call signature
    // **UPDATED** nnLinear to nnIdentity
    model = make_model(n_inputs, n_outputs, &params,
                       12, &nnRelu,     // bigger hidden layer for two outputs
                       12, &nnRelu,     // bigger hidden layer for two outputs
                       2, &nnIdentity, // output layer
                       0);
    if (!model) {
        nn_log(LOG_ERROR, "Failed to create model structure.");
        free(opt);
        return 1;
    }
    nn_log(LOG_INFO, "Model created.");

    // --- Train Model ---
    nn_log(LOG_INFO, "Starting training for %zu epochs...", epochs);
    bool train_success = model_train(model, inputs, targets, num_samples, epochs, train_batch_size);

    if (!train_success) {
        nn_log(LOG_ERROR, "Model training failed.");
        free_model(model);
        free(opt);
        return 1;
    }
    nn_log(LOG_INFO, "Training complete.");

    // --- Show Predictions ---
    printf("\n--- Final Predictions (Complex Square Root Fit) ---\n");
    pfloat prediction_buffer[2];

    for (size_t i = 0; i < num_samples; ++i) {
        const pfloat *input_val = inputs + (i * n_inputs);
        const pfloat *true_output = targets + (i * n_outputs);

        const pfloat *prediction = forward(model, input_val);

        if (prediction) {
            prediction_buffer[0] = prediction[0];
            prediction_buffer[1] = prediction[1];
            printf("Input: [%.2Lf] -> Target: (%.4Lf, %.4Lf), Prediction: (%.4Lf, %.4Lf)\n",
                   input_val[0],
                   true_output[0], true_output[1],
                   prediction_buffer[0], prediction_buffer[1]);
        } else {
            printf("Input: [%.2Lf] -> Prediction: FAILED\n", input_val[0]);
        }
    }
    printf("\n");

    // --- Final Cleanup ---
    nn_log(LOG_INFO, "Cleaning up resources...");
    free_model(model);
    free(opt);

    nn_log(LOG_INFO, "--- Testing Complete ---");
    return 0;
}
