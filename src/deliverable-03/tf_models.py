import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def create_nn_model_v1(embedding_dim: int, num_of_dense: int) -> Model:
    """
    Creates a neural network model that processes embedded user prompts using dense layers,
    concatenates it with function ratings, and passes through dense layers.

    Args:
        embedding_dim (int): Dimensionality of the embedding layer.
        num_of_dense (int): Number of dense layers before concatenation.

    Returns:
        Model: A compiled TensorFlow model.
    """
    # Text input - embedded vectors of specified embedding dimension
    text_input = Input(shape = (embedding_dim,), name = 'embedded_text_input')
    x = text_input

    # Dense layers for embedded text input
    # Start with enough neurons to have roughly embedding_dim^(1.5) squared neurons in initial layer
    # Restrict to 2 ** 12 = 4,096 neurons max for initial layer to keep model complexity
    # low enough for quick training and to use on small training datasets
    max_neurons = 2 ** min(np.round(np.log2(embedding_dim) * 1.5), 12)    
    
    # For every 2 layers, halve the number of neurons
    # This lets the model be very flexible for early layers and slowly become less flexible
    # This keeps the training relatively stochastic, which is helpful with low amounts of training data
    # If a layer has very few neurons, the model tends to plateau
    for i, _ in enumerate(range(num_of_dense)):
        # First layer uses max neurons, then halve the number of neurons every 2 layers
        divisor = 2 ** ((i + 1) // 2)
        num_neurons = max(1, int(max_neurons / divisor)) # Ensure integer neurons, minimum of 1
        x = Dense(num_neurons, activation = 'relu')(x)

    # Numeric input (func_rating) - one-hot encoded vector
    func_rating_input = Input(shape = (6,), name = 'func_rating_input')
    
    # Give some complexity to the function rating, but keep it simple with only one layer
    y = Dense(64, activation = 'relu')(func_rating_input)

    # Concatenate both paths, output is a classification task with 6 outputs
    concatenated = Concatenate()([x, y])
    output = Dense(6, activation = 'softmax', name = 'output')(concatenated)

    # Define and compile the model
    model = Model(inputs = [text_input, func_rating_input], outputs = output)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

def create_nn_model_v2(embedding_dim: int, num_of_dense: int,
                       train_size: int, batch_size: int,
                       l1_regularization: float = 0.0, l2_regularization: float = 0.0,
                       dense_step_rate: int = 1) -> Model:
    """
    Creates a neural network model that processes embedded user prompts using dense layers with regularization,
    concatenates it with function ratings, and passes through dense layers, using a LearningRateScheduler.

    Args:
        embedding_dim (int): Dimensionality of the embedding layer.
        num_of_dense (int): Number of dense layers before concatenation.
        train_size (int): Size of training samples used to calculate decay rate for LearningRateScheduler.
        batch_size (int): Size of batch used to calculate decay rate for LearningRateScheduler.
        l1_regularization (float): Level of L1 regularization to apply to Dense layers.
        l2_regularization (float): Level of L2 regularization to apply to Dense layers.
        dense_step_rate (int): Number of dense layers before halving the number of neurons.

    Returns:
        Model: A compiled TensorFlow model.
    """
    # Text input - embedded vectors of specified embedding dimension
    text_input = Input(shape = (embedding_dim,), name = 'embedded_text_input')
    x = text_input

    # Dense layers for embedded text input
    # Start with enough neurons to have roughly embedding_dim^(1.5) squared neurons in initial layer
    # Restrict to 2 ** 12 = 4,096 neurons max for initial layer to keep model complexity
    # low enough for quick training and to use on small training datasets
    max_neurons = 2 ** min(np.round(np.log2(embedding_dim) * 1.5), 12)
    
    # Add dense layers
    for i, _ in enumerate(range(num_of_dense)):
        # First layer uses max neurons, then halve the number of neurons every dense_step_rate layers
        divisor = 2 ** ((i + (dense_step_rate - 1)) // dense_step_rate)
        num_neurons = max(1, int(max_neurons / divisor)) # Ensure integer neurons, minimum of 1
        
        # Create dense layer with correct number of neurons and apply ElasticNet regularization
        x = Dense(num_neurons, kernel_regularizer = L1L2(l1_regularization, l2_regularization),
                  activation = 'relu')(x)

    # Numeric input (func_rating) - one-hot encoded vector
    func_rating_input = Input(shape = (6,), name = 'func_rating_input')
    
    # Give some complexity to the function rating, but keep it simple with only one layer
    y = Dense(64, activation = 'relu')(func_rating_input)

    # Concatenate both paths, output is a classification task with 6 outputs
    concatenated = Concatenate()([x, y])
    output = Dense(6, activation = 'softmax', name = 'output')(concatenated)

    # Define the model
    model = Model(inputs = [text_input, func_rating_input], outputs = output)
    
    # Compile the model using Adam optimizer and ExponentialDecay learning rate scheduler
    model.compile(optimizer = Adam(
        learning_rate = ExponentialDecay(
            initial_learning_rate = 0.001,
            decay_steps = int(train_size / batch_size),
            decay_rate = 0.96,
            staircase = True
    )), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model