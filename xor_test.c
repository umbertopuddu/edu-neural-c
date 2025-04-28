#include <stdio.h>
#include <stdlib.h>
#include "nn.h" // NN Framework header

int main(void) {
    nn_log(LOG_INFO, "--- NN Tester: XOR Problem ---");

    // --- Configuration ---
    size_t n_inputs = 2;
    size_t n_outputs = 1;
    size_t epochs = 100000;
    size_t train_batch_size = 4; // Batch size = dataset size
    pfloat learning_rate = 0.1L;
    int random_seed = 42;

    // --- XOR Data (Static) ---
    const pfloat inputs[] = {
        0.0L, 0.0L,
        0.0L, 1.0L,
        1.0L, 0.0L,
        1.0L, 1.0L
    };
    const pfloat targets[] = {
        0.0L,
        1.0L,
        1.0L,
        0.0L
    };
    size_t num_samples = sizeof(inputs) / (n_inputs * sizeof(pfloat));
    nn_log(LOG_INFO, "Using XOR dataset (%zu samples)", num_samples);

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

    // Hyperparameters
    params = (Params){
        .loss = &nnBCE,             // Binary Cross-Entropy suitable for XOR (and most logical tasks)
        .optimizer = opt,
        .seed = random_seed,
        .log_frequency_epochs = 10000, // Log less frequently for faster runs
        .randomize_batches = false    // Not needed for full-batch training
    };

    nn_log(LOG_INFO, "Creating model...");
    // Model: 2 -> 4 (ReLU) -> 1 (Sigmoid)
    model = make_model(n_inputs, n_outputs, &params,
                       4, &nnRelu,    // Hidden Layer
                       1, &nnSigmoid, // Output Layer (Sigmoid for probability-like output)
                       0);            // End of layers marker
    if (!model) {
        nn_log(LOG_ERROR, "Failed to create XOR model structure.");
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
    printf("\n--- Final Predictions (XOR) ---\n");
    pfloat prediction_buffer[1]; // Buffer for the single output

    for (size_t i = 0; i < num_samples; ++i) {
        const pfloat *current_input = inputs + (i * n_inputs);
        const pfloat current_target = targets[i * n_outputs]; // Direct access assuming n_outputs=1

        const pfloat *prediction = forward(model, current_input);

        if (prediction) {
            // Copy prediction to buffer for clarity, though direct use is possible
            prediction_buffer[0] = prediction[0];
            printf("Input: [%.1Lf, %.1Lf], Target: %.1Lf, Prediction: %.4Lf (Raw: %.4Lf)\n",
                   current_input[0], current_input[1], current_target,
                   (prediction_buffer[0] > 0.5L ? 1.0L : 0.0L), // Thresholded output (convert to binary)
                   prediction_buffer[0]);                      // Raw sigmoid output
        } else {
            printf("Input: [%.1Lf, %.1Lf], Target: %.1Lf, Prediction: FAILED\n",
                   current_input[0], current_input[1], current_target);
        }
    }
    printf("\n");
    success = true; // Mark success if predictions are shown

cleanup:
    nn_log(LOG_INFO, "Cleaning up resources...");
    free_model(model); // Safe to call on NULL
    free(opt);         // Safe to call on NULL
    nn_log(LOG_INFO, "--- XOR Testing Complete ---");
    return success ? 0 : 1; // Return 0 on success, 1 on failure
}
