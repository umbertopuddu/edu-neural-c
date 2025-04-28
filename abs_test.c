#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // For fabsl
#include "nn.h" // NN Framework header

int main(void) {
    nn_log(LOG_INFO, "--- NN Tester: Absolute Value Fit |x| ---");

    // --- Configuration ---
    size_t n_inputs = 1;
    size_t n_outputs = 1;
    size_t epochs = 80000;
    size_t train_batch_size = 10; // Using all samples per batch
    pfloat learning_rate = 0.01L;
    int random_seed = 456;

    // --- Absolute Value Data: y = |x| ---
    const pfloat inputs[] = {
        -2.0L, -1.5L, -1.0L, -0.5L, 0.0L, 0.5L, 1.0L, 1.5L, 2.0L, -0.25L
    };
    // Calculate targets directly
    pfloat targets[sizeof(inputs) / sizeof(inputs[0])];
    size_t num_samples = sizeof(inputs) / (n_inputs * sizeof(pfloat));
    for(size_t i = 0; i < num_samples; ++i) {
        targets[i] = fabsl(inputs[i]);
    }
    nn_log(LOG_INFO, "Using Absolute Value dataset (|x|, %zu samples)", num_samples);

    Model *model = NULL;
    Optimizer *opt = NULL;
    Params params;
    bool success = false;

    // --- Configure and Create Model ---
    opt = make_optimizer(learning_rate);
    if (!opt) {
        nn_log(LOG_ERROR, "Failed to create optimizer.");
        goto cleanup;
    }

    // Params struct initialization
    params = (Params){
        .loss = &nnMSE,             // Mean Squared Error for regression
        .optimizer = opt,
        .seed = random_seed,
        .log_frequency_epochs = 5000,
        .randomize_batches = true     // Shuffle even if batch size is full dataset
    };

    nn_log(LOG_INFO, "Creating model...");
    // Model: 1 -> 8 (ReLU) -> 8 (ReLU) -> 1 (Identity) - Added a layer
    model = make_model(n_inputs, n_outputs, &params,
                       8, &nnRelu,      // Hidden Layer 1
                       8, &nnRelu,      // Hidden Layer 2
                       1, &nnIdentity,  // Output Layer (Identity for regression)
                       0);              // End of layers marker
    if (!model) {
        nn_log(LOG_ERROR, "Failed to create model structure.");
        goto cleanup;
    }
    nn_log(LOG_INFO, "Model created successfully.");

    // --- Train Model ---
    nn_log(LOG_INFO, "Starting training for %zu epochs...", epochs);
    if (!model_train(model, inputs, targets, num_samples, epochs, train_batch_size)) {
        nn_log(LOG_ERROR, "Model training failed.");
        goto cleanup;
    }
    nn_log(LOG_INFO, "Training complete.");

    // --- Show Final Predictions ---
    printf("\n--- Final Predictions (Absolute Value Fit) ---\n");
    pfloat prediction_buffer[1]; // Buffer for the single output

    for (size_t i = 0; i < num_samples; ++i) {
        const pfloat *current_input = inputs + (i * n_inputs); // Pointer arithmetic for input
        const pfloat current_target = targets[i * n_outputs];

        const pfloat *prediction = forward(model, current_input);

        if (prediction) {
            prediction_buffer[0] = prediction[0];
            printf("Input: [%+.2Lf], Target: %.2Lf, Prediction: %.4Lf\n",
                   current_input[0], current_target, prediction_buffer[0]);
        } else {
            printf("Input: [%+.2Lf], Target: %.2Lf, Prediction: FAILED\n",
                   current_input[0], current_target);
        }
    }
    printf("\n");
    success = true;

cleanup:
    nn_log(LOG_INFO, "Cleaning up resources...");
    free_model(model);
    free(opt);
    nn_log(LOG_INFO, "--- Absolute Value Testing Complete ---");
    return success ? 0 : 1;
}