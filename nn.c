#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "nn.h"

// Check Numbers/Versions for storing formats
#define NN_PARAMS_CHECK 0x4E4E504D         // "NNPM" also referred to as magic (since the number is arbitrary, just to check dtype stored)
#define ENCODING_PARAMS_VERSION 0x00010001 // Currently 1.1
#define NN_FINAL_PARAMS_CHECK 0x4E4E4650   // "NNFP"
#define NN_FINAL_PARAMS_VERSION 0x00010001 // Currently 1.1

//----------------------------------------------------------------------
// Logging
//----------------------------------------------------------------------
void nn_log(LogLevel level, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    switch (level)
    {
    case LOG_INFO:
        fprintf(stderr, "[INFO] ");
        break;
    case LOG_WARN:
        fprintf(stderr, "[WARN] ");
        break;
    case LOG_ERROR:
        fprintf(stderr, "[ERROR] ");
        break;
    }
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

//----------------------------------------------------------------------
// Helper Functions
//----------------------------------------------------------------------
static pfloat rand_pfloat()
{
    return ((pfloat)rand() / (pfloat)RAND_MAX) * 2.0L - 1.0L; // Generates a random pfloat in the range [-1, 1]
}


static void seed_rng(int seed) // Set the seed for random weight generation
{
    static bool seeded = false;
    if (seed > 0)
    {
        srand((unsigned int)seed);
        seeded = true;
    }
    else if (!seeded)
    {
        srand((unsigned int)time(NULL));
        seeded = true;
    }
}

//----------------------------------------------------------------------
// Activation Functions & Derivatives
//----------------------------------------------------------------------

/* SIGMOID */

static pfloat sigmoid_forward(pfloat x)
{
    return 1.0L / (1.0L + expl(-x)); // Sigmoid function: 1 / (1 + e^(-x))
}

static pfloat sigmoid_backward(pfloat x)
{
    pfloat sig = sigmoid_forward(x);
    return sig * (1.0L - sig); // Derivative of sigmoid: sig * (1 - sig)
}

const nnFunc nnSigmoid = {sigmoid_forward, sigmoid_backward};

/* TANH */

static pfloat tanh_forward(pfloat x)
{
    return tanhl(x); // Hyperbolic tangent function: (e^x - e^(-x)) / (e^x + e^(-x)) (uses math.h implementation)
}

static pfloat tanh_backward(pfloat x)
{
    pfloat th = tanhl(x);
    return 1.0L - th * th; // Derivative of tanh: 1 - tanh^2
}

const nnFunc nnTanh = {tanh_forward, tanh_backward};

/* RELU */

static pfloat relu_forward(pfloat x) { return (x > 0.0L) ? x : 0.0L; }
static pfloat relu_backward(pfloat x) { return (x > 0.0L) ? 1.0L : 0.0L; }
const nnFunc nnRelu = {relu_forward, relu_backward};

/* IDENTITY (LINEAR) */

static pfloat identity_forward(pfloat x)
{
    return x;
}

static pfloat identity_backward(pfloat x)
{
    return 1.0L;
}

const nnFunc nnIdentity = {identity_forward, identity_backward};

//----------------------------------------------------------------------
// Loss Functions & Derivatives
//----------------------------------------------------------------------

/* MSE */

static pfloat mse_forward(const pfloat *y_pred, const pfloat *y_true, size_t n)
{
    pfloat s = 0.0L;
    for (size_t i = 0; i < n; ++i)
    {
        pfloat d = y_pred[i] - y_true[i];
        s += d * d;
    }
    return (n > 0) ? (s / (pfloat)n) : 0.0L;
}

static void mse_backward(const pfloat *y_pred, const pfloat *y_true, pfloat *grads, size_t n)
{
    if (n == 0)
    {
        return;
    }
    pfloat f = 2.0L / (pfloat)n;
    for (size_t i = 0; i < n; ++i)
    {
        grads[i] = f * (y_pred[i] - y_true[i]);
    }
}

const nnLoss nnMSE = {mse_forward, mse_backward};

/* BINARY CROSS-ENTROPY */

static pfloat bce_forward(const pfloat *y_pred, const pfloat *y_true, size_t n)
{
    pfloat l = 0.0L;
    const pfloat e = 1e-12L;
    if (n == 0)
        return 0.0L;
    for (size_t i = 0; i < n; ++i)
    {
        pfloat p = fmaxl(e, fminl(1.0L - e, y_pred[i]));
        l -= (y_true[i] * logl(p) + (1.0L - y_true[i]) * logl(1.0L - p));
    }
    return l / (pfloat)n;
}

static void bce_backward(const pfloat *y_pred, const pfloat *y_true, pfloat *grads, size_t n)
{
    const pfloat e = 1e-12L;
    if (n == 0)
        return;
    pfloat inv_n = 1.0L / (pfloat)n;
    for (size_t i = 0; i < n; ++i)
    {
        pfloat p = fmaxl(e, fminl(1.0L - e, y_pred[i]));
        pfloat d = p * (1.0L - p);
        if (fabsl(d) < e)
        {
            pfloat s = (p - y_true[i] > 0) ? 1.0L : -1.0L;
            grads[i] = s * 1.0L / e * inv_n;
        }
        else
        {
            grads[i] = inv_n * (p - y_true[i]) / d;
        }
    }
}

const nnLoss nnBCE = {bce_forward, bce_backward};

//----------------------------------------------------------------------
// Optimizer (make and free)
//----------------------------------------------------------------------
Optimizer *make_optimizer(pfloat learning_rate)
{
    Optimizer *opt = (Optimizer *)malloc(sizeof(Optimizer));
    if (!opt)
    {
        nn_log(LOG_ERROR, "Failed alloc Optimizer");
        return NULL;
    }
    opt->learning_rate = learning_rate;
    return opt;
}

//----------------------------------------------------------------------
// Constructors / Destructors
//----------------------------------------------------------------------

/* Neuron */

Neuron *make_neuron(size_t n_inputs, const nnFunc *act, bool zero_init, bool is_first_layer)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    if (!n)
    {
        nn_log(LOG_ERROR, "Failed alloc Neuron");
        return NULL;
    }
    n->n_inputs = n_inputs;
    n->act = act;
    n->inputs = NULL;
    if (n_inputs > 0)
    {
        if (!is_first_layer)
        {
            n->inputs = (Neuron **)malloc(n_inputs * sizeof(Neuron *));
            if (!n->inputs)
            {
                nn_log(LOG_ERROR, "Failed alloc Neuron inputs");
                free(n);
                return NULL;
            }
        }
        n->weights = (pfloat *)malloc(n_inputs * sizeof(pfloat));
        n->weight_grads = (pfloat *)malloc(n_inputs * sizeof(pfloat));
        if (!n->weights || !n->weight_grads)
        {
            nn_log(LOG_ERROR, "Failed alloc Neuron weights/grads");
            free(n->inputs);
            free(n->weights);
            free(n->weight_grads);
            free(n);
            return NULL;
        }
        if (zero_init)
        {
            for (size_t i = 0; i < n_inputs; ++i)
                n->weights[i] = 0.0L;
            n->bias = 0.0L;
        }
        else
        {
            pfloat scale = (n_inputs > 0) ? sqrtl(1.0L / (pfloat)n_inputs) : 1.0L;
            for (size_t i = 0; i < n_inputs; ++i)
                n->weights[i] = rand_pfloat() * scale;
            n->bias = 0.0L;
        }
    }
    return n;
}

void free_neuron(Neuron *n)
{
    if (!n)
    {
        return;
    }
    free(n->inputs);
    free(n->weights);
    free(n->weight_grads);
    free(n);
}

/* Layer */

Layer *make_layer(size_t n_neurons, size_t n_inputs_per_neuron, const nnFunc *act, bool is_first_layer)
{
    Layer *l = (Layer *) malloc(sizeof(Layer));
    if (!l)
    {
        nn_log(LOG_ERROR, "Failed alloc Layer");
        return NULL;
    }
    l->n_neurons = n_neurons;
    l->n_inputs_per_neuron = n_inputs_per_neuron;
    l->activation_func = act;
    if (n_neurons > 0)
    {
        l->neurons = (Neuron **)malloc(n_neurons * sizeof(Neuron *));
        if (!l->neurons)
        {
            nn_log(LOG_ERROR, "Failed alloc Layer neurons");
            free(l);
            return NULL;
        }
        for (size_t i = 0; i < n_neurons; ++i)
        {
            l->neurons[i] = make_neuron(n_inputs_per_neuron, act, false, is_first_layer);
            if (!l->neurons[i])
            {
                nn_log(LOG_ERROR, "Failed create neuron %zu in layer", i);
                for (size_t j = 0; j < i; ++j)
                {
                    free_neuron(l->neurons[j]);
                }
                free(l->neurons);
                free(l);
                return NULL;
            }
        }
    }
    return l;
}

void free_layer(Layer *l)
{
    if (!l)
    {
        return;
    }
    if (l->neurons)
    {
        for (size_t i = 0; i < l->n_neurons; ++i)
            free_neuron(l->neurons[i]);
        free(l->neurons);
    }
    free(l);
}

static bool connect_layers(Layer *prev_layer, Layer *curr_layer)
{
    if (!curr_layer || curr_layer->n_neurons == 0)
    {
        return true;
    }
    if (!prev_layer)
    {
        nn_log(LOG_ERROR, "connect_layers from NULL prev layer.");
        return false;
    }
    size_t n_prev_neurons = prev_layer->n_neurons;
    for (size_t i = 0; i < curr_layer->n_neurons; ++i)
    {
        Neuron *neuron = curr_layer->neurons[i];
        if (!neuron)
        {
            nn_log(LOG_ERROR, "connect_layers NULL current neuron %zu.", i);
            return false;
        }
        if (neuron->n_inputs != n_prev_neurons)
        {
            nn_log(LOG_ERROR, "connect_layers input count mismatch %zu vs %zu.", neuron->n_inputs, n_prev_neurons);
            return false;
        }
        if (neuron->n_inputs > 0 && !neuron->inputs)
        {
            nn_log(LOG_ERROR, "connect_layers NULL inputs array neuron %zu.", i);
            return false;
        }
        if(!prev_layer->neurons)
        {
            nn_log(LOG_ERROR, "connect_layers NULL prev layer neurons.");
            return false;
        }
        for (size_t j = 0; j < n_prev_neurons; ++j)
        {
            if (!prev_layer->neurons[j])
            {
                nn_log(LOG_ERROR, "connect_layers NULL prev neuron %zu.", j);
                return false;
            }
            neuron->inputs[j] = prev_layer->neurons[j];
        }
    }
    return true;
}

/* Model */

Model *make_model(size_t n_inputs, size_t n_outputs, const Params *params, ...)
{
    if (!params)
    {
        nn_log(LOG_ERROR, "make_model NULL params.");
        return NULL;
    }
    if (!params->loss || !params->optimizer)
    {
        nn_log(LOG_ERROR, "make_model requires loss and optimizer.");
        return NULL;
    }
    seed_rng(params->seed);
    Model *m = (Model *)malloc(sizeof(Model));
    if (!m)
    {
        nn_log(LOG_ERROR, "Failed alloc Model");
        return NULL;
    }
    m->params = params;
    m->n_inputs = n_inputs;   // Store n_inputs
    m->n_outputs = n_outputs; // Store n_outputs
    m->shuffled_indices = NULL; // Start batch shuffling with NULL
    m->num_shuffled_indices = 0;
    m->input_layer = (InputLayer *)malloc(sizeof(InputLayer));
    if (!m->input_layer)
    {
        nn_log(LOG_ERROR, "Failed alloc InputLayer");
        free(m);
        return NULL;
    }
    m->input_layer->n_inputs = n_inputs; // Use argument n_inputs
    if (n_inputs > 0)
    {
        m->input_layer->values = (pfloat *)malloc(n_inputs * sizeof(pfloat));
        if (!m->input_layer->values)
        {
            nn_log(LOG_ERROR, "Failed alloc InputLayer values");
            free(m->input_layer);
            free(m);
            return NULL;
        }
    }
    else
    {
        m->input_layer->values = NULL;
    }
    va_list args;
    va_start(args, params);
    size_t layer_count = 0;
    va_list count_args;
    va_copy(count_args, args);
    size_t first_layer_size_peek = va_arg(count_args, size_t);
    if (first_layer_size_peek != 0)
    {
        va_arg(count_args, const nnFunc *);
        layer_count++;
        while (va_arg(count_args, size_t) != 0)
        {
            va_arg(count_args, const nnFunc *);
            layer_count++;
        }
    }
    va_end(count_args);
    m->n_layers = layer_count;
    if (m->n_layers == 0)
    {
        nn_log(LOG_ERROR, "Model requires layers.");
        goto make_model_cleanup_input;
    }
    m->layers = (Layer **)calloc(m->n_layers, sizeof(Layer *));
    if (!m->layers)
    {
        nn_log(LOG_ERROR, "Failed alloc layers array");
        goto make_model_cleanup_input;
    }
    size_t prev_layer_size = n_inputs; // Start with model's n_inputs
    for (size_t i = 0; i < m->n_layers; ++i)
    {
        size_t layer_size = va_arg(args, size_t);
        const nnFunc *act = va_arg(args, const nnFunc *);
        if (layer_size == 0)
        {
            nn_log(LOG_ERROR, "Internal Error: Layer size 0.");
            goto make_model_cleanup_layers;
        }
        bool is_first = (i == 0);
        m->layers[i] = make_layer(layer_size, prev_layer_size, act, is_first);
        if (!m->layers[i])
        {
            nn_log(LOG_ERROR, "Failed create layer %zu (size %zu)", i, layer_size);
            goto make_model_cleanup_layers;
        }
        if (!is_first)
        {
            if (!connect_layers(m->layers[i - 1], m->layers[i]))
            {
                nn_log(LOG_ERROR, "Failed connect layer %zu to %zu", i - 1, i);
                goto make_model_cleanup_layers;
            }
        }
        prev_layer_size = layer_size;
    }
    va_end(args);
    m->output_layer = m->layers[m->n_layers - 1];
    // Use model's n_outputs for verification
    if (m->output_layer->n_neurons != m->n_outputs)
    {
        nn_log(LOG_ERROR, "Final layer size (%zu) != model n_outputs (%zu).", m->output_layer->n_neurons, m->n_outputs);
        goto make_model_cleanup_layers;
    }
    m->output_buffer_size = m->n_outputs; // Use model's n_outputs
    if (m->output_buffer_size > 0)
    {
        m->output_buffer = (pfloat *)malloc(m->output_buffer_size * sizeof(pfloat));
        if (!m->output_buffer)
        {
            nn_log(LOG_ERROR, "Failed alloc output buffer");
            goto make_model_cleanup_layers;
        }
    }
    else
    {
        m->output_buffer = NULL;
    }
    return m;
make_model_cleanup_layers:
    if (m && m->layers)
    {
        for (size_t i = 0; i < m->n_layers; ++i)
            free_layer(m->layers[i]);
        free(m->layers);
    }
make_model_cleanup_input:
    if (m && m->input_layer)
    {
        free(m->input_layer->values);
        free(m->input_layer);
    }
    free(m);
    va_end(args);
    return NULL;
}
void free_model(Model *m)
{
    if (!m)
        return;
    if (m->layers)
    {
        for (size_t i = 0; i < m->n_layers; ++i)
            free_layer(m->layers[i]);
        free(m->layers);
    }
    if (m->input_layer)
    {
        free(m->input_layer->values);
        free(m->input_layer);
    }
    free(m->output_buffer);
    free(m->shuffled_indices);
    free(m);
}

void free_finalized_model(FinalizedModel *fm)
{
    if (!fm)
        return;
    if (fm->layers)
    {
        for (size_t i = 0; i < fm->n_layers; ++i)
        {
            free(fm->layers[i].weights);
            free(fm->layers[i].biases);
        }
        free(fm->layers);
    }
    free(fm);
}

/* Finalised Model */

FinalizedModel *model_finalize(const Model *m)
{
    if (!m || !m->layers || m->n_layers == 0)
    {
        nn_log(LOG_ERROR, "Cannot finalize NULL or invalid model.");
        return NULL;
    }
    FinalizedModel *fm = (FinalizedModel *)malloc(sizeof(FinalizedModel));
    if (!fm)
    {
        nn_log(LOG_ERROR, "Failed alloc FinalizedModel struct.");
        return NULL;
    }
    fm->n_inputs = m->n_inputs;   // Get from model struct
    fm->n_outputs = m->n_outputs; // Get from model struct
    fm->n_layers = m->n_layers;
    fm->layers = (FinalizedLayer *)calloc(fm->n_layers, sizeof(FinalizedLayer));
    if (!fm->layers)
    {
        nn_log(LOG_ERROR, "Failed alloc FinalizedLayer array.");
        free(fm);
        return NULL;
    }
    for (size_t i = 0; i < fm->n_layers; ++i)
    {
        Layer *orig_layer = m->layers[i];
        if (!orig_layer || !orig_layer->neurons)
        {
            nn_log(LOG_ERROR, "Original layer %zu or neurons NULL during finalization.", i);
            free_finalized_model(fm);
            return NULL;
        }
        fm->layers[i].n_neurons = orig_layer->n_neurons;
        fm->layers[i].n_inputs = orig_layer->n_inputs_per_neuron;
        fm->layers[i].activation = orig_layer->activation_func;
        if (fm->layers[i].n_neurons == 0)
        {
            continue;
        }
        size_t num_weights = fm->layers[i].n_neurons * fm->layers[i].n_inputs;
        if (num_weights > 0)
        {
            fm->layers[i].weights = (pfloat *)malloc(num_weights * sizeof(pfloat));
            if (!fm->layers[i].weights)
            {
                nn_log(LOG_ERROR, "Failed alloc weights finalized layer %zu.", i);
                free_finalized_model(fm);
                return NULL;
            }
        }
        else
        {
            fm->layers[i].weights = NULL;
        }
        fm->layers[i].biases = (pfloat *)malloc(fm->layers[i].n_neurons * sizeof(pfloat));
        if (!fm->layers[i].biases)
        {
            nn_log(LOG_ERROR, "Failed alloc biases finalized layer %zu.", i);
            free_finalized_model(fm);
            return NULL;
        }
        for (size_t j = 0; j < fm->layers[i].n_neurons; ++j)
        {
            Neuron *orig_neuron = orig_layer->neurons[j];
            if (!orig_neuron)
            {
                nn_log(LOG_ERROR, "Original neuron %zu layer %zu NULL during finalization.", j, i);
                free_finalized_model(fm);
                return NULL;
            }
            fm->layers[i].biases[j] = orig_neuron->bias;
            if (num_weights > 0)
            {
                if (!orig_neuron->weights)
                {
                    nn_log(LOG_ERROR, "Original neuron %zu layer %zu weights NULL.", j, i);
                    free_finalized_model(fm);
                    return NULL;
                }
                memcpy(&fm->layers[i].weights[j * fm->layers[i].n_inputs], orig_neuron->weights, fm->layers[i].n_inputs * sizeof(pfloat));
            }
        }
    }
    return fm;
}

pfloat *finalized_model_predict(const FinalizedModel *fm, const pfloat *input_vec, pfloat *output_buffer)
{
    if (!fm || !input_vec || !output_buffer)
    {
        nn_log(LOG_ERROR, "finalized_model_predict NULL arguments.");
        return NULL;
    }
    if (fm->n_layers == 0)
    {
        nn_log(LOG_ERROR, "finalized_model_predict 0 layers.");
        return NULL;
    }
    pfloat *layer_input = NULL;
    pfloat *layer_output = NULL;
    size_t layer_input_size = fm->n_inputs;
    if (layer_input_size > 0)
    {
        layer_input = (pfloat *)malloc(layer_input_size * sizeof(pfloat));
        if (!layer_input)
        {
            nn_log(LOG_ERROR, "Failed alloc initial input buffer predict.");
            return NULL;
        }
        memcpy(layer_input, input_vec, layer_input_size * sizeof(pfloat));
    }
    for (size_t i = 0; i < fm->n_layers; ++i)
    {
        const FinalizedLayer *layer = &fm->layers[i];
        size_t layer_output_size = layer->n_neurons;
        if (layer_output_size > 0)
        {
            layer_output = (pfloat *)malloc(layer_output_size * sizeof(pfloat));
            if (!layer_output)
            {
                nn_log(LOG_ERROR, "Failed alloc output buffer layer %zu predict.", i);
                free(layer_input);
                return NULL;
            }
        }
        else
        {
            layer_output = NULL;
            if (i < fm->n_layers - 1 && fm->layers[i + 1].n_inputs != 0)
            {
                nn_log(LOG_ERROR, "Layer %zu has 0 neurons, next layer expects inputs.", i);
                free(layer_input);
                free(layer_output);
                return NULL;
            }
        }
        for (size_t j = 0; j < layer->n_neurons; ++j)
        {
            pfloat weighted_sum = layer->biases[j];
            size_t weight_start_index = j * layer->n_inputs;
            for (size_t k = 0; k < layer->n_inputs; ++k)
            {
                if (!layer_input)
                {
                    nn_log(LOG_ERROR, "NULL layer_input predict (L%zu, N%zu).", i, j);
                    free(layer_input);
                    free(layer_output);
                    return NULL;
                }
                if (!layer->weights)
                {
                    nn_log(LOG_ERROR, "NULL layer weights predict (L%zu, N%zu).", i, j);
                    free(layer_input);
                    free(layer_output);
                    return NULL;
                }
                weighted_sum += layer->weights[weight_start_index + k] * layer_input[k];
            }
            if (layer->activation && layer->activation->forward)
            {
                layer_output[j] = layer->activation->forward(weighted_sum);
            }
            else
            {
                layer_output[j] = weighted_sum;
            }
        }
        free(layer_input);
        layer_input = layer_output;
        layer_input_size = layer_output_size;
        layer_output = NULL;
    }
    if (layer_input_size != fm->n_outputs)
    {
        nn_log(LOG_ERROR, "Final layer output size (%zu) mismatch model n_outputs (%zu).", layer_input_size, fm->n_outputs);
        free(layer_input);
        return NULL;
    }
    if (layer_input)
    {
        memcpy(output_buffer, layer_input, fm->n_outputs * sizeof(pfloat));
        free(layer_input);
    }
    else if (fm->n_outputs > 0)
    {
        nn_log(LOG_ERROR, "Final layer output NULL but n_outputs > 0.");
        return NULL;
    }
    return output_buffer;
}

//----------------------------------------------------------------------
// Training Functions
//----------------------------------------------------------------------
void model_zero_grads(Model *m)
{
    if (!m)
        return;
    for (size_t i = 0; i < m->n_layers; ++i)
    {
        Layer *layer = m->layers[i];
        if (!layer)
        {
            continue;
        }
        for (size_t j = 0; j < layer->n_neurons; ++j)
        {
            Neuron *neuron = layer->neurons[j];
            if (!neuron)
            {
                continue;
            }
            neuron->bias_grad = 0.0L;
            neuron->grad = 0.0L;
            if (neuron->weight_grads)
            {
                memset(neuron->weight_grads, 0, neuron->n_inputs * sizeof(pfloat));
            }
        }
    }
    m->grads_ready = false;
}

const pfloat *forward(Model *m, const pfloat *input_vec)
{
    if (!m)
    {
        nn_log(LOG_ERROR, "forward NULL model.");
        return NULL;
    }
    if (!m->input_layer)
    {
        nn_log(LOG_ERROR, "forward NULL input layer.");
        return NULL;
    }
    size_t n_inputs = m->n_inputs;
    if (n_inputs > 0)
    {
        if (!input_vec)
        {
            nn_log(LOG_ERROR, "forward NULL input_vec.");
            return NULL;
        }
        if (!m->input_layer->values)
        {
            nn_log(LOG_ERROR, "forward NULL input values buffer.");
            return NULL;
        }
        memcpy(m->input_layer->values, input_vec, n_inputs * sizeof(pfloat));
    }
    else
    {
        if (input_vec)
        {
            nn_log(LOG_WARN, "forward input_vec provided for 0-input model.");
        }
    }
    for (size_t i = 0; i < m->n_layers; ++i)
    {
        Layer *current_layer = m->layers[i];
        if (!current_layer)
        {
            nn_log(LOG_ERROR, "forward NULL layer %zu.", i);
            return NULL;
        }
        bool is_first = (i == 0);
        for (size_t j = 0; j < current_layer->n_neurons; ++j)
        {
            Neuron *neuron = current_layer->neurons[j];
            if (!neuron)
            {
                nn_log(LOG_ERROR, "forward NULL neuron L%zu N%zu.", i, j);
                return NULL;
            }
            pfloat weighted_sum = neuron->bias;
            for (size_t k = 0; k < neuron->n_inputs; ++k)
            {
                if (!neuron->weights)
                {
                    nn_log(LOG_ERROR, "forward NULL weights L%zu N%zu.", i, j);
                    return NULL;
                }
                pfloat input_val;
                if (is_first)
                {
                    if (k >= n_inputs)
                    {
                        nn_log(LOG_ERROR, "forward input index %zu OOB %zu.", k, n_inputs);
                        return NULL;
                    }
                    input_val = m->input_layer->values[k];
                }
                else
                {
                    if (!neuron->inputs || !neuron->inputs[k])
                    {
                        nn_log(LOG_ERROR, "forward NULL input ptr L%zu N%zu k%zu.", i, j, k);
                        return NULL;
                    }
                    input_val = neuron->inputs[k]->value;
                }
                weighted_sum += neuron->weights[k] * input_val;
            }
            neuron->pre_activation_value = weighted_sum;
            neuron->value = (neuron->act && neuron->act->forward) ? neuron->act->forward(weighted_sum) : weighted_sum;
        }
    }
    if (!m->output_layer)
    {
        nn_log(LOG_ERROR, "forward NULL output layer ptr.");
        return NULL;
    }
    if (m->output_layer->n_neurons != m->output_buffer_size)
    {
        nn_log(LOG_ERROR, "forward output layer/buffer size mismatch.");
        return NULL;
    }
    if (m->output_buffer)
    {
        for (size_t i = 0; i < m->output_buffer_size; ++i)
        {
            if (!m->output_layer->neurons || !m->output_layer->neurons[i])
            {
                nn_log(LOG_ERROR, "forward NULL output neuron %zu.", i);
                return NULL;
            }
            m->output_buffer[i] = m->output_layer->neurons[i]->value;
        }
    }
    else if (m->output_buffer_size > 0)
    {
        nn_log(LOG_ERROR, "forward output buffer NULL despite size > 0.");
        return NULL;
    }
    return m->output_buffer;
}

void backward(Model *m, const pfloat *y_true)
{
    if (!m)
    {
        nn_log(LOG_ERROR, "backward NULL model.");
        return;
    }
    if (!y_true)
    {
        nn_log(LOG_ERROR, "backward NULL y_true.");
        return;
    }
    if (!m->output_layer || !m->params || !m->params->loss || !m->params->loss->backward || m->n_layers == 0)
    {
        nn_log(LOG_ERROR, "backward Invalid model state.");
        return;
    }
    size_t n_outputs = m->n_outputs;
    if (n_outputs == 0)
    {
        m->grads_ready = true;
        return;
    }
    if (!m->output_buffer)
    {
        nn_log(LOG_ERROR, "backward NULL output buffer.");
        return;
    }
    pfloat *dLoss_dOutput = (pfloat *)malloc(n_outputs * sizeof(pfloat));
    if (!dLoss_dOutput)
    {
        nn_log(LOG_ERROR, "Failed alloc backward dLoss/dOutput");
        return;
    }
    m->params->loss->backward(m->output_buffer, y_true, dLoss_dOutput, n_outputs);
    Layer *layer = m->output_layer;
    for (size_t i = 0; i < layer->n_neurons; ++i)
    {
        Neuron *neuron = layer->neurons[i];
        if (!neuron)
        {
            nn_log(LOG_ERROR, "backward output NULL neuron %zu.", i);
            goto backward_cleanup;
        }
        pfloat dL_da = dLoss_dOutput[i];
        pfloat da_dz = (neuron->act && neuron->act->backward) ? neuron->act->backward(neuron->pre_activation_value) : 1.0L;
        pfloat dL_dz = dL_da * da_dz;
        neuron->grad = dL_dz;
        neuron->bias_grad += dL_dz;
        if (neuron->weight_grads)
        {
            for (size_t k = 0; k < neuron->n_inputs; ++k)
            {
                if (!neuron->inputs || !neuron->inputs[k])
                {
                    nn_log(LOG_ERROR, "backward output NULL input %zu for neuron %zu.", k, i);
                    goto backward_cleanup;
                }
                neuron->weight_grads[k] += dL_dz * neuron->inputs[k]->value;
            }
        }
        else if (neuron->n_inputs > 0)
        {
            nn_log(LOG_ERROR, "backward output weight_grads NULL neuron %zu.", i);
            goto backward_cleanup;
        }
    }
    for (int layer_idx = m->n_layers - 2; layer_idx >= 0; --layer_idx)
    {
        Layer *current_layer = m->layers[layer_idx];
        Layer *next_layer = m->layers[layer_idx + 1];
        if (!current_layer || !next_layer)
        {
            nn_log(LOG_ERROR, "backward NULL layer %d or %d.", layer_idx, layer_idx + 1);
            goto backward_cleanup;
        }
        bool is_first = (layer_idx == 0);
        for (size_t j = 0; j < current_layer->n_neurons; ++j)
        {
            Neuron *neuron = current_layer->neurons[j];
            if (!neuron)
            {
                nn_log(LOG_ERROR, "backward hidden NULL neuron L%d N%zu.", layer_idx, j);
                goto backward_cleanup;
            }
            pfloat sum_weighted_grads = 0.0L;
            for (size_t k = 0; k < next_layer->n_neurons; ++k)
            {
                Neuron *next_neuron = next_layer->neurons[k];
                if (!next_neuron)
                {
                    nn_log(LOG_ERROR, "backward NULL next_neuron L%d N%zu.", layer_idx + 1, k);
                    goto backward_cleanup;
                }
                if (j < next_neuron->n_inputs)
                {
                    if (!next_neuron->weights)
                    {
                        nn_log(LOG_ERROR, "backward NULL weights next_neuron L%d N%zu.", layer_idx + 1, k);
                        goto backward_cleanup;
                    }
                    sum_weighted_grads += next_neuron->grad * next_neuron->weights[j];
                }
                else
                {
                    nn_log(LOG_ERROR, "backward index %zu OOB next_neuron %zu inputs (%zu).", j, k, next_neuron->n_inputs);
                    goto backward_cleanup;
                }
            }
            pfloat dL_da = sum_weighted_grads;
            pfloat da_dz = (neuron->act && neuron->act->backward) ? neuron->act->backward(neuron->pre_activation_value) : 1.0L;
            pfloat dL_dz = dL_da * da_dz;
            neuron->grad = dL_dz;
            neuron->bias_grad += dL_dz;
            if (neuron->weight_grads)
            {
                for (size_t k = 0; k < neuron->n_inputs; ++k)
                {
                    pfloat input_val;
                    if (is_first)
                    {
                        if (!m->input_layer || !m->input_layer->values)
                        {
                            nn_log(LOG_ERROR, "backward input values NULL.");
                            goto backward_cleanup;
                        }
                        if (k >= m->n_inputs)
                        {
                            nn_log(LOG_ERROR, "backward input index %zu OOB %zu.", k, m->n_inputs);
                            goto backward_cleanup;
                        }
                        input_val = m->input_layer->values[k];
                    }
                    else
                    {
                        if (!neuron->inputs || !neuron->inputs[k])
                        {
                            nn_log(LOG_ERROR, "backward hidden NULL input L%d N%zu k%zu.", layer_idx, j, k);
                            goto backward_cleanup;
                        }
                        input_val = neuron->inputs[k]->value;
                    }
                    neuron->weight_grads[k] += dL_dz * input_val;
                }
            }
            else if (neuron->n_inputs > 0)
            {
                nn_log(LOG_ERROR, "backward hidden weight_grads NULL L%d N%zu.", layer_idx, j);
                goto backward_cleanup;
            }
        }
    }
    m->grads_ready = true;
backward_cleanup:
    free(dLoss_dOutput);
}

void model_apply_gradients(Model *m, size_t batch_size)
{
    if (!m || !m->params || !m->params->optimizer)
    {
        nn_log(LOG_ERROR, "model_apply_gradients NULL model/params/optimizer.");
        return;
    }
    if (batch_size == 0)
    {
        nn_log(LOG_ERROR, "model_apply_gradients batch_size is 0.");
        return;
    }
    if (!m->grads_ready)
    {
        nn_log(LOG_WARN, "model_apply_gradients called but gradients not ready.");
        return;
    }
    pfloat lr = m->params->optimizer->learning_rate;
    if (lr <= 0)
    {
        nn_log(LOG_WARN, "model_apply_gradients non-positive LR (%Lf).", lr);
        return;
    }
    pfloat adjusted_lr = lr / (pfloat)batch_size;
    for (size_t i = 0; i < m->n_layers; ++i)
    {
        Layer *layer = m->layers[i];
        if (!layer)
            continue;
        for (size_t j = 0; j < layer->n_neurons; ++j)
        {
            Neuron *neuron = layer->neurons[j];
            if (!neuron)
                continue;
            neuron->bias -= adjusted_lr * neuron->bias_grad;
            if (neuron->n_inputs > 0)
            {
                if (neuron->weights && neuron->weight_grads)
                {
                    for (size_t k = 0; k < neuron->n_inputs; ++k)
                    {
                        neuron->weights[k] -= adjusted_lr * neuron->weight_grads[k];
                    }
                }
            }
        }
    }
    m->grads_ready = false; // Gradients have been used
}

static pfloat model_process_batch(Model *m, const pfloat *all_inputs, const pfloat *all_targets, const size_t *batch_indices, size_t num_samples_in_batch)
{
    if (!m || !m->params || !all_inputs || !all_targets || !batch_indices)
    {
        nn_log(LOG_ERROR, "model_process_batch: NULL arguments.");
        return -1.0L;
    }
    if (num_samples_in_batch == 0)
        return 0.0L;
    if (!m->params->loss || !m->params->loss->forward)
    {
        nn_log(LOG_ERROR, "model_process_batch: Missing loss function.");
        return -1.0L;
    }
    model_zero_grads(m);
    pfloat total_batch_loss = 0.0L;
    size_t n_inputs = m->n_inputs;
    size_t n_outputs = m->n_outputs;
    bool error_occurred = false;
    for (size_t i = 0; i < num_samples_in_batch; ++i)
    {
        size_t sample_index = batch_indices[i];
        const pfloat *current_input = all_inputs + (sample_index * n_inputs);
        const pfloat *current_target = all_targets + (sample_index * n_outputs);
        const pfloat *predictions = forward(m, current_input);
        if (!predictions)
        {
            nn_log(LOG_ERROR, "model_process_batch: Forward failed sample index %zu.", sample_index);
            error_occurred = true;
            break;
        }
        total_batch_loss += m->params->loss->forward(predictions, current_target, n_outputs);
        backward(m, current_target);
    }
    if (!error_occurred)
    {
        if (m->grads_ready)
        {
            model_apply_gradients(m, num_samples_in_batch);
        }
        else
        {
            nn_log(LOG_WARN, "model_process_batch: No gradients accumulated.");
        }
        return total_batch_loss / (pfloat)num_samples_in_batch;
    }
    else
    {
        model_zero_grads(m);
        return -1.0L;
    }
}

static void shuffle_indices(size_t *array, size_t n)
{
    if (n > 1)
    {
        for (size_t i = n - 1; i > 0; i--)
        {
            size_t j = (size_t)rand() % (i + 1);
            size_t temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}

bool model_train(Model *model, const pfloat *all_inputs, const pfloat *all_targets, size_t num_samples, size_t epochs, size_t batch_size)
{
    if (!model || !model->params || !all_inputs || !all_targets)
    {
        nn_log(LOG_ERROR, "model_train: NULL arguments.");
        return false;
    }
    if (num_samples == 0)
    {
        nn_log(LOG_WARN, "model_train: num_samples is 0.");
        return true;
    }
    if (epochs == 0)
    {
        nn_log(LOG_WARN, "model_train: epochs is 0.");
        return true;
    }
    if (batch_size == 0)
    {
        nn_log(LOG_ERROR, "model_train: batch_size cannot be 0.");
        return false;
    }
    nn_log(LOG_INFO, "Starting training: %zu epochs, %zu samples, batch size %zu, randomize %s", epochs, num_samples, batch_size, model->params->randomize_batches ? "true" : "false");
    if (model->params->randomize_batches)
    {
        if (model->num_shuffled_indices != num_samples)
        {
            free(model->shuffled_indices);
            model->shuffled_indices = (size_t *)malloc(num_samples * sizeof(size_t));
            if (!model->shuffled_indices)
            {
                nn_log(LOG_ERROR, "model_train: Failed alloc shuffle buffer.");
                return false;
            }
            model->num_shuffled_indices = num_samples;
            for (size_t i = 0; i < num_samples; ++i)
                model->shuffled_indices[i] = i;
        }
    }
    else
    {
        free(model->shuffled_indices);
        model->shuffled_indices = NULL;
        model->num_shuffled_indices = 0;
    }
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        if (model->params->randomize_batches && model->shuffled_indices)
        {
            shuffle_indices(model->shuffled_indices, num_samples);
        }
        for (size_t i = 0; i < num_samples; i += batch_size)
        {
            size_t batch_end_offset = i + batch_size;
            if (batch_end_offset > num_samples)
            {
                batch_end_offset = num_samples;
            }
            size_t current_batch_size = batch_end_offset - i;
            if (current_batch_size > 0)
            {
                size_t *current_batch_indices = NULL;
                if (model->params->randomize_batches && model->shuffled_indices)
                {
                    current_batch_indices = model->shuffled_indices + i;
                }
                else
                {
                    size_t *sequential_indices = (size_t *)malloc(current_batch_size * sizeof(size_t));
                    if (!sequential_indices)
                    {
                        nn_log(LOG_ERROR, "Failed alloc temp indices");
                        return false;
                    }
                    for (size_t k = 0; k < current_batch_size; ++k)
                        sequential_indices[k] = i + k;
                    current_batch_indices = sequential_indices;
                }
                pfloat batch_loss = model_process_batch(model, all_inputs, all_targets, current_batch_indices, current_batch_size);
                if (!model->params->randomize_batches)
                {
                    free(current_batch_indices);
                }
                if (batch_loss < 0.0L)
                {
                    nn_log(LOG_ERROR, "Training failed during batch processing epoch %zu", epoch);
                    return false;
                }
            }
        }
        if (model->params->log_frequency_epochs > 0 && ((epoch + 1) % model->params->log_frequency_epochs == 0 || epoch == 0 || epoch == epochs - 1))
        {
            pfloat avg_loss = model_calculate_average_loss(model, all_inputs, all_targets, num_samples);
            if (avg_loss < 0.0L)
            {
                nn_log(LOG_ERROR, "Failed calculate average loss epoch %zu", epoch);
            }
            else
            {
                nn_log(LOG_INFO, ">>> Epoch %zu/%zu, Avg Full Loss: %Lf <<<", epoch + 1, epochs, avg_loss);
            }
        }
    }
    nn_log(LOG_INFO, "Training finished successfully.");
    return true;
}

pfloat model_calculate_average_loss(Model *model, const pfloat *all_inputs, const pfloat *all_targets, size_t num_samples)
{
    if (!model || !model->params || !model->params->loss || !model->params->loss->forward)
    {
        nn_log(LOG_ERROR, "model_calculate_average_loss: Invalid model or missing loss function.");
        return -1.0L;
    }
    if (num_samples == 0)
        return 0.0L;
    pfloat total_loss = 0.0L;
    size_t n_inputs = model->n_inputs;
    size_t n_outputs = model->n_outputs;
    for (size_t i = 0; i < num_samples; ++i)
    {
        const pfloat *current_input = all_inputs + (i * n_inputs);
        const pfloat *current_target = all_targets + (i * n_outputs);
        const pfloat *predictions = forward(model, current_input);
        if (!predictions)
        {
            nn_log(LOG_ERROR, "Forward pass failed during loss calculation sample %zu.", i);
            return -1.0L;
        }
        total_loss += model->params->loss->forward(predictions, current_target, n_outputs);
    }
    return total_loss / (pfloat)num_samples;
}

//----------------------------------------------------------------------
// Saving / Loading Model and Finalised Model (only parameters)
//----------------------------------------------------------------------
static bool write_data(const void *data, size_t size, size_t count, FILE *fp, const char *filename, const char *data_name)
{
    if (fwrite(data, size, count, fp) != count)
    {
        nn_log(LOG_ERROR, "Failed write %s to '%s'.", data_name, filename);
        return false;
    }
    return true;
}

static bool read_data(void *data, size_t size, size_t count, FILE *fp, const char *filename, const char *data_name)
{
    if (fread(data, size, count, fp) != count)
    {
        if (feof(fp))
        {
            nn_log(LOG_ERROR, "EOF reading %s from '%s'.", data_name, filename);
        }
        else
        {
            nn_log(LOG_ERROR, "Failed read %s from '%s'.", data_name, filename);
        }
        return false;
    }
    return true;
}

bool model_save_params(const Model *m, const char *filename)
{
    if (!m || !filename)
    {
        nn_log(LOG_ERROR, "model_save_params: Invalid arguments.");
        return false;
    }
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        nn_log(LOG_ERROR, "model_save_params: Failed open '%s' write.", filename);
        return false;
    }
    bool success = true;
    uint32_t magic = NN_PARAMS_CHECK;
    uint32_t version = ENCODING_PARAMS_VERSION;
    size_t n_inputs = m->n_inputs;
    size_t n_outputs = m->n_outputs;
    size_t n_layers = m->n_layers;
    success &= write_data(&magic, sizeof(magic), 1, fp, filename, "magic");
    success &= write_data(&version, sizeof(version), 1, fp, filename, "version");
    success &= write_data(&n_layers, sizeof(n_layers), 1, fp, filename, "n_layers");
    success &= write_data(&n_inputs, sizeof(n_inputs), 1, fp, filename, "n_inputs (verify)");
    success &= write_data(&n_outputs, sizeof(n_outputs), 1, fp, filename, "n_outputs (verify)");
    if (success)
    {
        for (size_t i = 0; i < n_layers; ++i)
        {
            Layer *layer = m->layers[i];
            if (!layer)
            {
                nn_log(LOG_ERROR, "model_save_params: NULL layer %zu.", i);
                success = false;
                break;
            }
            size_t neurons = layer->n_neurons;
            size_t inputs_per_neuron = layer->n_inputs_per_neuron;
            success &= write_data(&neurons, sizeof(neurons), 1, fp, filename, "neurons");
            success &= write_data(&inputs_per_neuron, sizeof(inputs_per_neuron), 1, fp, filename, "inputs_per_neuron");
            if (!success)
                break;
            for (size_t j = 0; j < neurons; ++j)
            {
                Neuron *neuron = layer->neurons[j];
                if (!neuron)
                {
                    nn_log(LOG_ERROR, "model_save_params: NULL neuron L%zu N%zu.", i, j);
                    success = false;
                    break;
                }
                if (neuron->n_inputs != inputs_per_neuron)
                {
                    nn_log(LOG_ERROR, "model_save_params: Mismatch n_inputs N%zu L%zu.", j, i);
                    success = false;
                    break;
                }
                success &= write_data(&neuron->bias, sizeof(neuron->bias), 1, fp, filename, "bias");
                if (!success)
                    break;
                if (neuron->n_inputs > 0)
                {
                    if (!neuron->weights)
                    {
                        nn_log(LOG_ERROR, "model_save_params: NULL weights N%zu L%zu.", j, i);
                        success = false;
                        break;
                    }
                    success &= write_data(neuron->weights, sizeof(pfloat), neuron->n_inputs, fp, filename, "weights");
                    if (!success)
                        break;
                }
            }
            if (!success)
                break;
        }
    }
    fclose(fp);
    if (!success)
    {
        nn_log(LOG_ERROR, "Model params saving failed for '%s'.", filename);
        remove(filename);
        return false;
    }
    nn_log(LOG_INFO, "Model params saved successfully to '%s'.", filename);
    return true;
}

bool model_load_params(Model *m, const char *filename)
{
    if (!m || !filename)
    {
        nn_log(LOG_ERROR, "model_load_params: Requires a non-NULL model and filename.");
        return false;
    }
    if (!m->layers || m->n_layers == 0)
    {
        nn_log(LOG_ERROR, "model_load_params: Provided model has no layers allocated.");
        return false;
    }
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        nn_log(LOG_ERROR, "model_load_params: Failed open '%s'.", filename);
        return false;
    }
    bool success = true;
    uint32_t magic = 0, version = 0;
    size_t file_n_layers = 0;
    size_t file_n_inputs = 0, file_n_outputs = 0;
    success &= read_data(&magic, sizeof(magic), 1, fp, filename, "magic");
    success &= read_data(&version, sizeof(version), 1, fp, filename, "version");
    if (!success || magic != NN_PARAMS_CHECK)
    {
        nn_log(LOG_ERROR, "model_load_params: Invalid magic number '%s'.", filename);
        fclose(fp);
        return false;
    }
    if (version > ENCODING_PARAMS_VERSION)
    {
        nn_log(LOG_WARN, "model_load_params: File version %X newer than lib %X.", version, ENCODING_PARAMS_VERSION);
    }
    success &= read_data(&file_n_layers, sizeof(file_n_layers), 1, fp, filename, "n_layers");
    success &= read_data(&file_n_inputs, sizeof(file_n_inputs), 1, fp, filename, "n_inputs (verify)");
    success &= read_data(&file_n_outputs, sizeof(file_n_outputs), 1, fp, filename, "n_outputs (verify)");
    if (!success)
    {
        fclose(fp);
        return false;
    }
    if (file_n_layers != m->n_layers)
    {
        nn_log(LOG_ERROR, "model_load_params: Layer count mismatch (file %zu vs model %zu) '%s'.", file_n_layers, m->n_layers, filename);
        fclose(fp);
        return false;
    }
    if (m->n_inputs != file_n_inputs)
    {
        nn_log(LOG_ERROR, "model_load_params: Input count mismatch (file %zu vs model %zu) '%s'.", file_n_inputs, m->n_inputs, filename);
        fclose(fp);
        return false;
    }
    if (m->n_outputs != file_n_outputs)
    {
        nn_log(LOG_ERROR, "model_load_params: Output count mismatch (file %zu vs model %zu) '%s'.", file_n_outputs, m->n_outputs, filename);
        fclose(fp);
        return false;
    }
    for (size_t i = 0; i < m->n_layers; ++i)
    {
        Layer *layer = m->layers[i];
        if (!layer)
        {
            nn_log(LOG_ERROR, "model_load_params: Provided model has NULL layer %zu.", i);
            success = false;
            break;
        }
        size_t file_neurons = 0, file_inputs_per_neuron = 0;
        success &= read_data(&file_neurons, sizeof(file_neurons), 1, fp, filename, "neurons");
        success &= read_data(&file_inputs_per_neuron, sizeof(file_inputs_per_neuron), 1, fp, filename, "inputs_per_neuron");
        if (!success)
            break;
        if (file_neurons != layer->n_neurons)
        {
            nn_log(LOG_ERROR, "model_load_params: Neuron count mismatch L%zu (file %zu vs model %zu) '%s'.", i, file_neurons, layer->n_neurons, filename);
            success = false;
            break;
        }
        if (file_inputs_per_neuron != layer->n_inputs_per_neuron)
        {
            nn_log(LOG_ERROR, "model_load_params: Inputs per neuron mismatch L%zu (file %zu vs model %zu) '%s'.", i, file_inputs_per_neuron, layer->n_inputs_per_neuron, filename);
            success = false;
            break;
        }
        for (size_t j = 0; j < layer->n_neurons; ++j)
        {
            Neuron *neuron = layer->neurons[j];
            if (!neuron)
            {
                nn_log(LOG_ERROR, "model_load_params: Provided model NULL neuron L%zu N%zu.", i, j);
                success = false;
                break;
            }
            if (neuron->n_inputs != layer->n_inputs_per_neuron)
            {
                nn_log(LOG_ERROR, "model_load_params: Provided model neuron input mismatch L%zu N%zu.", i, j);
                success = false;
                break;
            }
            success &= read_data(&neuron->bias, sizeof(neuron->bias), 1, fp, filename, "bias");
            if (!success)
                break;
            if (neuron->n_inputs > 0)
            {
                if (!neuron->weights)
                {
                    nn_log(LOG_ERROR, "model_load_params: Provided model NULL weights L%zu N%zu.", i, j);
                    success = false;
                    break;
                }
                success &= read_data(neuron->weights, sizeof(pfloat), neuron->n_inputs, fp, filename, "weights");
                if (!success)
                    break;
            }
        }
        if (!success)
            break;
    }
    fclose(fp);
    if (!success)
    {
        nn_log(LOG_ERROR, "Model param loading failed for '%s'. Model state might be inconsistent.", filename);
        return false;
    }
    nn_log(LOG_INFO, "Model params loaded successfully into provided model from '%s'.", filename);
    return true;
}

bool finalized_model_save_params(const FinalizedModel *fm, const char *filename)
{
    if (!fm || !filename)
    {
        nn_log(LOG_ERROR, "finalized_model_save_params: Invalid arguments.");
        return false;
    }
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        nn_log(LOG_ERROR, "finalized_model_save_params: Failed open '%s'.", filename);
        return false;
    }
    bool success = true;
    uint32_t magic = NN_FINAL_PARAMS_CHECK;
    uint32_t version = NN_FINAL_PARAMS_VERSION;
    size_t n_inputs = fm->n_inputs;
    size_t n_outputs = fm->n_outputs;
    size_t n_layers = fm->n_layers;
    success &= write_data(&magic, sizeof(magic), 1, fp, filename, "magic");
    success &= write_data(&version, sizeof(version), 1, fp, filename, "version");
    success &= write_data(&n_layers, sizeof(n_layers), 1, fp, filename, "n_layers");
    success &= write_data(&n_inputs, sizeof(n_inputs), 1, fp, filename, "n_inputs (verify)");
    success &= write_data(&n_outputs, sizeof(n_outputs), 1, fp, filename, "n_outputs (verify)");
    if (success)
    {
        for (size_t i = 0; i < n_layers; ++i)
        {
            const FinalizedLayer *layer = &fm->layers[i];
            size_t neurons = layer->n_neurons;
            size_t inputs_per_layer = layer->n_inputs;
            success &= write_data(&neurons, sizeof(neurons), 1, fp, filename, "f neurons");
            success &= write_data(&inputs_per_layer, sizeof(inputs_per_layer), 1, fp, filename, "f inputs");
            if (!success)
                break;
            if (neurons > 0)
            {
                if (!layer->biases)
                {
                    nn_log(LOG_ERROR, "finalized_model_save_params: NULL biases L%zu.", i);
                    success = false;
                    break;
                }
                success &= write_data(layer->biases, sizeof(pfloat), neurons, fp, filename, "f biases");
                if (!success)
                    break;
            }
            size_t num_weights = neurons * inputs_per_layer;
            if (num_weights > 0)
            {
                if (!layer->weights)
                {
                    nn_log(LOG_ERROR, "finalized_model_save_params: NULL weights L%zu.", i);
                    success = false;
                    break;
                }
                success &= write_data(layer->weights, sizeof(pfloat), num_weights, fp, filename, "f weights");
                if (!success)
                    break;
            }
        }
    }
    fclose(fp);
    if (!success)
    {
        nn_log(LOG_ERROR, "Finalized model param saving failed for '%s'.", filename);
        remove(filename);
        return false;
    }
    nn_log(LOG_INFO, "Finalized model params saved successfully to '%s'.", filename);
    return true;
}

bool finalized_model_load_params(FinalizedModel *fm, const char *filename)
{
    if (!fm || !filename)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Requires non-NULL model and filename.");
        return false;
    }
    if (!fm->layers && fm->n_layers > 0)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Provided model has no layers allocated.");
        return false;
    }
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Failed open '%s'.", filename);
        return false;
    }
    bool success = true;
    uint32_t magic = 0, version = 0;
    size_t file_n_layers = 0;
    size_t file_n_inputs = 0, file_n_outputs = 0;
    success &= read_data(&magic, sizeof(magic), 1, fp, filename, "magic");
    success &= read_data(&version, sizeof(version), 1, fp, filename, "version");
    if (!success || magic != NN_FINAL_PARAMS_CHECK)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Invalid magic '%s'.", filename);
        fclose(fp);
        return false;
    }
    if (version > NN_FINAL_PARAMS_VERSION)
    {
        nn_log(LOG_WARN, "finalized_model_load_params: File version %X newer than lib %X.", version, NN_FINAL_PARAMS_VERSION);
    }
    success &= read_data(&file_n_layers, sizeof(file_n_layers), 1, fp, filename, "n_layers");
    success &= read_data(&file_n_inputs, sizeof(file_n_inputs), 1, fp, filename, "n_inputs (verify)");
    success &= read_data(&file_n_outputs, sizeof(file_n_outputs), 1, fp, filename, "n_outputs (verify)");
    if (!success)
    {
        fclose(fp);
        return false;
    }
    if (file_n_layers != fm->n_layers)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Layer count mismatch (file %zu vs model %zu) '%s'.", file_n_layers, fm->n_layers, filename);
        fclose(fp);
        return false;
    }
    if (file_n_inputs != fm->n_inputs)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Input count mismatch (file %zu vs model %zu) '%s'.", file_n_inputs, fm->n_inputs, filename);
        fclose(fp);
        return false;
    }
    if (file_n_outputs != fm->n_outputs)
    {
        nn_log(LOG_ERROR, "finalized_model_load_params: Output count mismatch (file %zu vs model %zu) '%s'.", file_n_outputs, fm->n_outputs, filename);
        fclose(fp);
        return false;
    }
    for (size_t i = 0; i < fm->n_layers; ++i)
    {
        FinalizedLayer *layer = &fm->layers[i];
        size_t file_neurons = 0, file_inputs_per_layer = 0;
        success &= read_data(&file_neurons, sizeof(file_neurons), 1, fp, filename, "f neurons");
        success &= read_data(&file_inputs_per_layer, sizeof(file_inputs_per_layer), 1, fp, filename, "f inputs");
        if (!success)
            break;
        if (file_neurons != layer->n_neurons)
        {
            nn_log(LOG_ERROR, "finalized_model_load_params: Neuron count mismatch L%zu (file %zu vs model %zu) '%s'.", i, file_neurons, layer->n_neurons, filename);
            success = false;
            break;
        }
        if (file_inputs_per_layer != layer->n_inputs)
        {
            nn_log(LOG_ERROR, "finalized_model_load_params: Input count mismatch L%zu (file %zu vs model %zu) '%s'.", i, file_inputs_per_layer, layer->n_inputs, filename);
            success = false;
            break;
        }
        if (layer->n_neurons > 0)
        {
            if (!layer->biases)
            {
                nn_log(LOG_ERROR, "finalized_model_load_params: Provided model NULL biases L%zu.", i);
                success = false;
                break;
            }
            success &= read_data(layer->biases, sizeof(pfloat), layer->n_neurons, fp, filename, "f biases");
            if (!success)
                break;
        }
        size_t num_weights = layer->n_neurons * layer->n_inputs;
        if (num_weights > 0)
        {
            if (!layer->weights)
            {
                nn_log(LOG_ERROR, "finalized_model_load_params: Provided model NULL weights L%zu.", i);
                success = false;
                break;
            }
            success &= read_data(layer->weights, sizeof(pfloat), num_weights, fp, filename, "f weights");
            if (!success)
                break;
        }
    }
    fclose(fp);
    if (!success)
    {
        nn_log(LOG_ERROR, "Finalized model param loading failed for '%s'.", filename);
        return false;
    }
    nn_log(LOG_INFO, "Finalized model params loaded successfully into provided model from '%s'.", filename);
    return true;
}
