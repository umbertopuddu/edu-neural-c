#ifndef NN_H
#define NN_H

// Include standard headers FIRST
#include <stddef.h>  // For size_t
#include <stdbool.h> // For bool type
#include <stdarg.h>  // For va_list, va_start, etc.
#include <stdint.h>  // For uint32_t

// Floating-point type used throughout the network
typedef long double pfloat;

/*----------------------------------------------------------------------
 * Logging
 *--------------------------------------------------------------------*/
typedef enum
{
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;
void nn_log(LogLevel level, const char *format, ...);

/*----------------------------------------------------------------------
 * Activation Functions (Scalar Ops)
 *--------------------------------------------------------------------*/
typedef pfloat (*func)(pfloat);
typedef struct nnFunc
{
    func forward;
    func backward;
} nnFunc;

extern const nnFunc nnSigmoid, nnTanh, nnRelu, nnIdentity, nnMultiply;

/*----------------------------------------------------------------------
 * Loss Functions
 *--------------------------------------------------------------------*/
typedef pfloat (*loss_forward_fn)(const pfloat *y_pred, const pfloat *y_true, size_t n);
typedef void (*loss_backward_fn)(const pfloat *y_pred, const pfloat *y_true, pfloat *grads, size_t n);
typedef struct nnLoss
{
    loss_forward_fn forward;
    loss_backward_fn backward;
} nnLoss;

extern const nnLoss nnMSE, nnBCE;

/*----------------------------------------------------------------------
 * Optimizer
 *--------------------------------------------------------------------*/
typedef struct optimizer
{
    pfloat learning_rate;
} Optimizer;

Optimizer *make_optimizer(pfloat learning_rate);

/*----------------------------------------------------------------------
 * Core Network Structures (Trainable Model)
 *--------------------------------------------------------------------*/
typedef struct neuron Neuron;
typedef struct layer Layer;
typedef struct ilayer InputLayer;
typedef struct model Model;
typedef struct params Params;
typedef Layer *HiddenLayer, *OutputLayer;

struct neuron
{
    size_t n_inputs;
    Neuron **inputs;
    pfloat *weights;
    pfloat bias;
    pfloat value;
    pfloat pre_activation_value;
    pfloat grad;
    pfloat bias_grad;
    pfloat *weight_grads;
    const nnFunc *act;
};

struct layer
{
    size_t n_neurons;
    Neuron **neurons;
    size_t n_inputs_per_neuron;
    const nnFunc *activation_func;
};

struct ilayer
{
    size_t n_inputs;
    pfloat *values;
};

struct params
{
    const nnLoss *loss;          // Loss function to use
    Optimizer *optimizer;        // Optimizer configuration (SGD and minibatch SGD for now)
    int seed;                    // Seed for random weight initialization (<= 0 uses time dependent randomisation)
    size_t log_frequency_epochs; // Log full dataset loss every N epochs in model_train (0 disables logging)
    bool randomize_batches;      // Shuffle dataset indices each epoch in model_train ?
};
struct model
{
    const Params *params; // Pointer to user-managed training parameters
    size_t n_inputs;      // Number of inputs
    size_t n_outputs;     // Number of outputs
    InputLayer *input_layer;
    OutputLayer output_layer;
    size_t n_layers;
    Layer **layers;
    bool grads_ready;
    pfloat *output_buffer;
    size_t output_buffer_size; // output_buffer_size should == n_outputs

    // For minibatch SGD
    size_t *shuffled_indices;
    size_t num_shuffled_indices;
};

/*----------------------------------------------------------------------
 * Finalized Model Structure
 *--------------------------------------------------------------------*/
typedef struct finalized_layer
{
    size_t n_neurons;
    size_t n_inputs;
    pfloat *weights;
    pfloat *biases;
    const nnFunc *activation;
} FinalizedLayer;

typedef struct finalized_model
{
    size_t n_inputs;
    size_t n_outputs;
    size_t n_layers;
    FinalizedLayer *layers;
} FinalizedModel;

/*----------------------------------------------------------------------
 * Constructors / Destructors
 *--------------------------------------------------------------------*/
Neuron *make_neuron(size_t n_inputs, const nnFunc *act, bool zero_init, bool is_first_layer);
void free_neuron(Neuron *n);
Layer *make_layer(size_t n_neurons, size_t n_inputs_per_neuron, const nnFunc *act, bool is_first_layer);
void free_layer(Layer *l);

/**
 * @brief Creates and initializes a complete trainable neural network model.
 * @param n_inputs Number of input features the model expects.
 * @param n_outputs Number of output neurons the model should produce.
 * @param params Pointer to the Params struct containing training configuration (loss, optimizer, etc.). Model stores this pointer but doesn't own it or the optimizer within it (must be free separately).
 * @param ... Variable arguments specifying layer sizes and activation functions (size_t num_neurons, const nnFunc *activation), terminated by 0 (to signal no additional layers in the model).
 * @return Pointer to the newly created Model, or NULL on failure.
 */
Model *make_model(size_t n_inputs, size_t n_outputs, const Params *params, ...);

void free_model(Model *m);
void free_finalized_model(FinalizedModel *fm);

/*----------------------------------------------------------------------
 * Model Finalization & Inference
 *--------------------------------------------------------------------*/
FinalizedModel *model_finalize(const Model *m);
pfloat *finalized_model_predict(const FinalizedModel *fm, const pfloat *input_vec, pfloat *output_buffer);

/*----------------------------------------------------------------------
 * Training Functions
 *--------------------------------------------------------------------*/
void model_zero_grads(Model *m);
const pfloat *forward(Model *m, const pfloat *input_vec);
void backward(Model *m, const pfloat *y_true);
void model_apply_gradients(Model *m, size_t batch_size); // Internal gradient application step (SGD)

bool model_train(Model *model, const pfloat *all_inputs, const pfloat *all_targets,
                 size_t num_samples, size_t epochs, size_t batch_size);

pfloat model_calculate_average_loss(Model *model, const pfloat *all_inputs, const pfloat *all_targets, size_t num_samples); // Loss over entire dataset (not per batch)

/*----------------------------------------------------------------------
 * Saving / Loading (Weights & Biases Only)
 *--------------------------------------------------------------------*/
bool model_save_params(const Model *m, const char *filename);
bool model_load_params(Model *m, const char *filename);
bool finalized_model_save_params(const FinalizedModel *fm, const char *filename);
bool finalized_model_load_params(FinalizedModel *fm, const char *filename);

#endif // NN_H
