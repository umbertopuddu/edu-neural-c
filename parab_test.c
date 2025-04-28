#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For target function
#include "nn.h" // NN Framework header

// Target function: y = x^2
static pfloat target_function(pfloat x) {
    return x * x;
}

int main(void) {
    nn_log(LOG_INFO, "--- NN Tester: Parabola (y=x^2) Problem ---");

    // --- Configuration ---
    size_t n_inputs = 1;
    size_t n_outputs = 1;
    size_t num_samples = 101;     // Generate 101 points from -2 to 2
    size_t epochs = 70000;
    size_t train_batch_size = 16;
    pfloat learning_rate = 0.0005L;
    int random_seed = 555;
    pfloat range_min = -2.0L;
    pfloat range_max = 2.0L;

    Model *model = NULL;
    Optimizer *opt = NULL;
    Params params;
    pfloat *inputs = NULL;
    pfloat *targets = NULL;
    bool success = false;

    // --- Generate Data ---
    nn_log(LOG_INFO, "Generating %zu samples for y = x^2 in range [%.1Lf, %.1Lf]",
           num_samples, range_min, range_max);
    inputs = (pfloat*)malloc(num_samples * n_inputs * sizeof(pfloat));
    targets = (pfloat*)malloc(num_samples * n_outputs * sizeof(pfloat));
    if (!inputs || !targets) {
        nn_log(LOG_ERROR, "Failed to allocate data arrays");
        goto cleanup;
    }

    for(size_t i = 0; i < num_samples; ++i) {
        pfloat x = range_min + (range_max - range_min) * (pfloat)i / (pfloat)(num_samples - 1);
        inputs[i * n_inputs] = x; // Assuming n_inputs = 1
        targets[i * n_outputs] = target_function(x); // Assuming n_outputs = 1
    }
    nn_log(LOG_INFO, "Data generation complete.");

    // --- Configure and Create Model ---
    opt = make_optimizer(learning_rate);
    if (!opt) {
        nn_log(LOG_ERROR, "Failed to create optimizer.");
        goto cleanup;
    }

    // Hyperparameters
    params = (Params){
        .loss = &nnMSE,             // Mean Squared Error for regression
        .optimizer = opt,
        .seed = random_seed,
        .log_frequency_epochs = 10000,
        .randomize_batches = true
    };

    nn_log(LOG_INFO, "Creating model...");
    // Model: 1 -> 16 (ReLU) -> 16 (ReLU) -> 1 (Identity) - Slightly larger
    model = make_model(n_inputs, n_outputs, &params,
                       16, &nnRelu,      // Hidden Layer 1
                       16, &nnRelu,      // Hidden Layer 2
                       1, &nnIdentity,  // Output Layer
                       0);
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

    // --- Show Predictions on Test Points ---
    printf("\n--- Final Predictions (y = x^2) ---\n");
    pfloat prediction_buffer[1];
    const pfloat test_points[] = {-2.5L, -2.0L, -1.0L, 0.0L, 1.0L, 2.0L, 2.5L}; // Test cases (some outside training range)
    size_t num_test_points = sizeof(test_points) / sizeof(test_points[0]);

    for (size_t i = 0; i < num_test_points; ++i) {
        pfloat input_val = test_points[i];
        pfloat target_val = target_function(input_val);

        // Pass address of input_val since forward expects a pointer
        const pfloat *prediction = forward(model, &input_val);

        if (prediction) {
             prediction_buffer[0] = prediction[0];
             printf("Input: %+.2Lf, Target: %+.4Lf, Prediction: %+.4Lf\n",
                   input_val, target_val, prediction_buffer[0]);
        } else {
             printf("Input: %+.2Lf, Target: %+.4Lf, Prediction: FAILED\n",
                   input_val, target_val);
        }
    }
    printf("\n");
    success = true;

cleanup:
    nn_log(LOG_INFO, "Cleaning up resources...");
    free_model(model);
    free(opt);
    free(inputs);
    free(targets);
    nn_log(LOG_INFO, "--- Parabola Testing Complete ---");
    return success ? 0 : 1;
}