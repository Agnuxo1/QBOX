"""
## QBOX  +1BIT 
**Francisco Angulo de Lafuente**  
**May 23, 2024**

https://github.com/Agnuxo1

**QBOX: Quantum-Inspired Box for Neural Network Exploration in Ray Tracing 3D Cube Environment**

QBOX is a pioneering platform designed to explore and enhance the capabilities of neural networks 
through the integration of quantum-inspired computational models and ray tracing techniques within 
a dynamic 3D cube environment. This innovative framework facilitates the development, training, 
and deployment of advanced artificial intelligence algorithms, leveraging the unique features 
of quantum mechanics to drive breakthroughs in computational efficiency, pattern recognition, 
and predictive analysis. 

By encapsulating complex neural structures in a virtual 'box,' QBOX offers researchers 
a controlled environment to experiment with cutting-edge AI methodologies, fostering 
the creation of next-generation solutions in areas such as computer vision, 
graphics rendering, and data science.


**Abstract:**

This program simulates a 3D photonic neural processor designed to perform addition with two-bit numbers. 
It leverages principles of Ray Tracing for efficient signal
propagation between photonic neurons, mimicking biological neural
communication. Each neuron operates on a +1-bit activation scheme, emitting
light pulses to transmit information.

The network utilizes a novel approach of neuronal collaboration, where
groups of neighboring neurons work together to achieve higher computational
precision, similar to having 2 or 3 bits of capacity. The processor is
organized into specialized areas to handle specific tasks, further
enhancing its efficiency.

The learning process employs principles of Hebbian learning, promoting biological plausibility and
energy efficiency. This project explores the potential of photonic
neural processors for developing energy-efficient, high-performance AI
systems, particularly suited for edge devices and resource-constrained
scenarios. This implementation uses CUDA for accelerated ray tracing calculations
on NVIDIA GPUs, drastically improving performance. The distance matrix calculation
is further optimized using shared memory for enhanced speed.
"""


import os  # For file management
import time  # For timekeeping
import math  # For math functions such as sqrt
import numpy as np  # Import NumPy for array handling
import cupy as cp  # Import CuPy for GPU acceleration
import psutil  # For system monitoring (CPU, Memory)
import GPUtil  # For GPU monitoring
import random  # For random number generation
import ray  # For parallelization
from numba import cuda  # For GPU acceleration and Ray Tracing
from deap import base, creator, tools, algorithms  # For global parameter optimization
import threading
import CUBE_REPORT  # Import the custom report module
from scipy.stats import entropy  # For calculating information entropy

# --- Visualization Flag ---
Neuron_Activity = 0  # 0: Disabled, 1: Enabled

# --- Parameters ---

CUBE_SIZE = 2  # Cube size: 2x2x2
MAX_RAY_DISTANCE = 8  # Maximum distance a light ray can travel within the cube
ATTENUATION_FACTOR = 0.3  # Factor to reduce signal intensity over distance
FLASH_INTERVAL = 0.1  # Time interval between light pulses
NUM_PULSES = 4  # Number of pulses per flash
CARRIER_COUNT = 64 # Number of carriers for OFDM
FLASH_COUNT = 100  # Total number of flashes for training
TRAINING_FLASH_X = 1  # Number of processes for training flashes:
# 1: Traditional training (all neurons train together)
# 2: Two parallel training groups
# 3: Three parallel training groups
# 4: Four parallel training groups (by neuron class: Level1, Level2, Level3, Director+Communicators)
BIT_PRECISION = 2  # Bit precision for intensity and encoding
BATCH_SIZE = 16  # Batch size for batch normalization
OUTPUT_NEURON_COUNT = 2  # Number of output neurons (for binary encoding)
NEURON_COUNT = 99  # Neuron count: 1 Director + 3 Communicators + 94 other neurons + 1 Bias Neuron "ADJUST ACTIVE NEURON COUNT TO LEVELS"
LEVELS_X = 2 # Number of levels in the optical cube (0: only Director, 1, 2, 3... to infinity)

# --- Grid Parameters ---
GRID_DIMENSIONS = np.array([10, 10, 10])  # Dimensions of the spatial grid
CELL_SIZE = 1  # Size of each grid cell
MAX_NEIGHBORS_PER_CELL = 50  # Define the maximum number of neighbors a cell can have

# --- Neuron Parameters ---

MIN_OPACITY_THRESHOLD = 0.1  # Minimum threshold for opacity of the neuron
MAX_OPACITY_THRESHOLD = 0.7  # Maximum threshold for opacity of the neuron
MIN_REFLECTANCE = 0.1  # Minimum reflectance for the neuron
MAX_REFLECTANCE = 0.9  # Maximum reflectance for the neuron
MIN_EMISSION_INTENSITY = (
    0.1  # Minimum emission ray intensity for the neuron
)
MAX_EMISSION_INTENSITY = (
    0.9  # Maximum emission ray intensity for the neuron
)
MIN_INTENSITY_DECAY = 0.1  # Minimum intensity reflection decay for the neuron
MAX_INTENSITY_DECAY = 0.9  # Maximum intensity reflection decay for the neuron
MIN_LEARNING_RATE = 0.009  # Increased from 0.05
MAX_LEARNING_RATE = 0.05  # Increased from 0.1
MIN_REFLECTANCE_DECAY = 0.1
MAX_REFLECTANCE_DECAY = 0.9
NOISE_AMPLITUDE = 0.003  # Random luminosity noise amplitude

# --- Global Learning Parameters ---

RECURRENCE_FACTOR = 0.2  # Controls influence of previous intensity
INTENSITY_DECAY = 0.5  # Rate of intensity decay
REFLECTANCE_DECAY = 0.03  # Rate of reflectance decay
HEBBIAN_LEARNING_RATE = 0.001  # Learning rate for Hebbian learning
REWARD_RATE = 1  # Reward for correct responses (Increased from 0.1)
PENALTY_RATE = 0.1  # Penalty for incorrect responses

# --- Fine-Tuning Parameters ---

FINE_TUNE_FLASHES = 50  # Number of flashes to use during fine-tuning
FINE_TUNE_ITERATIONS = 3  # Number of fine-tuning iterations
FINE_TUNE_STEP_SIZE = 0.05  # Step size for parameter adjustments
AUTO_FINE_TUNING_X = 0  # Number of processes for fine-tuning (0: disabled, >1: enabled)

# --- Association Parameters ---
ASSOCIATION_EVALUATION_INTERVAL = 10  # Evaluate association every 10 iterations
TARGET_PRECISION = 8  # Desired precision in bits for associations
ENTROPY_THRESHOLD = 1.0  # Entropy threshold for dissolving associations

# --- Watchdog Parameters ---

WATCHDOG_INTENSITY = 0  # Intensity level of the watchdog (0: disabled, 1-9: increasing intensity)
WATCHDOG_CHECK_DURATION = 5  # Duration (in seconds) for the watchdog to perform checks
WATCHDOG_REST_DURATION = 5  # Duration (in seconds) for the watchdog to rest between checks

# Import CUBE_REPORT (handle ImportError if module not found)
try:
    import CUBE_REPORT
except ImportError:
    print(
        "Warning: CUBE_REPORT module not found. Reporting functionality will be limited."
    )

# --- Grid Class ---
class Grid:
    """
    Represents the spatial grid used for efficient neighbor search during ray tracing.

    Attributes:
        dimensions (numpy.array): Dimensions of the grid in 3D space.
        cell_size (float): Size of each cell in the grid.
        grid (dict): Dictionary storing neuron indices in each cell. Key is the cell index,
        and value is a list of neuron indices.
    """

    def __init__(self, dimensions, cell_size):
        self.dimensions = dimensions
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_index(self, position):
        """
        Calculates the index of the grid cell based on a given position.

        Args:
            position (numpy.array): The 3D coordinates (x, y, z) of a point.

        Returns:
            tuple: A tuple representing the (x, y, z) index of the grid cell containing the point.
        """
        return tuple((position // self.cell_size).astype(int))

    def add_neuron(self, neuron, index):
        """
        Adds a neuron to the corresponding grid cell based on its position.

        Args:
            neuron (Neuron): The Neuron object to add to the grid.
            index (int): The index of the neuron in the neural_cube_data list.
        """
        position = np.array([neuron.x, neuron.y, neuron.z])
        cell_index = self._get_cell_index(position)
        if cell_index not in self.grid:
            self.grid[cell_index] = []
        self.grid[cell_index].append(index)  # Add the neuron's index to the grid cell

    def get_neighbors(self, neuron, neurons):
        """
        Retrieves neighboring neurons of a given neuron from the grid.

        Args:
            neuron (Neuron): The neuron for which to find neighbors.
            neurons (list): The list of all neurons in the cube.

        Returns:
            list: A list of Neuron objects that are neighbors of the given neuron.
        """
        position = np.array([neuron.x, neuron.y, neuron.z])
        cell_index = self._get_cell_index(position)
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    neighboring_cell = (
                        cell_index[0] + i,
                        cell_index[1] + j,
                        cell_index[2] + k,
                    )
                    if neighboring_cell in self.grid:
                        neighbors.extend(self.grid[neighboring_cell])
        return [
            neurons[i] for i in neighbors
        ]  # Return the actual Neuron objects based on their indices


# --- Neuron Classes ---
class Neuron:
    """
    Represents a neuron in the optical cube.

    Attributes:
        name (str): Unique name of the neuron.
        x (float): x-coordinate of the neuron in the cube.
        y (float): y-coordinate of the neuron in the cube.
        z (float): z-coordinate of the neuron in the cube.
        neuron_type (str): Type of neuron ('Director', 'Communicator', 'Level1', 'Level2', 'Level3', 'Output', 'Bias').
        opacity_threshold (float): Light intensity threshold for the neuron to become active.
        received_intensity (float): Current light intensity received by the neuron.
        previous_intensity (float): Light intensity received in the previous timestep.
        active (bool): True if the neuron is currently active, False otherwise.
        reflectance (float): Proportion of light the neuron reflects.
        emission_intensity (float): Intensity of light emitted when the neuron is active.
        intensity_decay (float): Rate at which received light intensity decays over time.
        learning_rate (float): Learning rate for adjusting reflectance.
        reflectance_decay (float): Rate at which reflectance decays over time.
        memory (int): Not used currently, placeholder for future implementations.
        bias (float): Bias value added to the received intensity.
        precision (int): Current precision of the neuron in bits (initial precision is set to 2).
        associated_neurons (list): List of neurons associated with this neuron.
        meta_neuron (MetaNeuron): Reference to the MetaNeuron if this neuron is part of an association.
    """

    def __init__(
        self,
        name,
        x,
        y,
        z,
        neuron_type,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
        initial_precision=2,
    ):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.neuron_type = neuron_type
        self.opacity_threshold = opacity_threshold
        self.received_intensity = 0.0
        self.previous_intensity = 0.0
        self.active = False
        self.reflectance = reflectance
        self.emission_intensity = emission_intensity
        self.intensity_decay = intensity_decay
        self.learning_rate = random.uniform(MIN_LEARNING_RATE, MAX_LEARNING_RATE)
        self.reflectance_decay = random.uniform(
            MIN_REFLECTANCE_DECAY, MAX_REFLECTANCE_DECAY
        )
        self.memory = 3
        self.bias = bias
        self.precision = initial_precision
        self.associated_neurons = []
        self.meta_neuron = None

    def __repr__(self):
        return f"{self.name} ({self.x}, {self.y}, {self.z}, {self.opacity_threshold:.2f}, {self.received_intensity:.2f}, {self.active}, {self.reflectance:.2f})"

    def sigmoid(self, x):
        """Sigmoid activation function (currently not used)."""
        return 1 / (1 + np.exp(-x))

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function."""
        return max(alpha * x, x)

    def he_initialization(self):
        """He initialization for reflectance (currently not used)."""
        return random.gauss(0, math.sqrt(2 / NEURON_COUNT))

    def update_activation(self):
        """Activates the neuron if received intensity exceeds the opacity threshold."""
        self.active = (
            self.leaky_relu(self.received_intensity - self.opacity_threshold) > 0
        )

    def limit_intensity(self):
        """Limits the received intensity to a reasonable range."""
        self.received_intensity = min(max(self.received_intensity, -10), 10)

    def adjust_reflectance(self, delta):
        """Adjusts the reflectance based on learning."""
        self.reflectance += self.learning_rate * delta
        self.reflectance = max(-2, min(self.reflectance, 2))

    def apply_reflectance_decay(self):
        """Applies reflectance decay over time."""
        self.reflectance *= 1 - self.reflectance_decay

    def update_intensity(self, neural_cube_intensity, i, dampening_factor=0.9):
        """Updates the received intensity based on propagation and other factors."""
        self.previous_intensity = self.received_intensity
        self.received_intensity = (
            0.8 * self.previous_intensity
            + 0.2
            * (
                neural_cube_intensity[i]
                + (self.emission_intensity * self.reflectance if self.active else 0)
            )
        )
        self.received_intensity += self.bias + np.random.normal(0, 0.05)
        self.received_intensity *= dampening_factor
        self.received_intensity = min(max(self.received_intensity, -10), 10)
        self.update_activation()


# Define specialized neuron classes (empty for now - inherit from Neuron)
class Director(Neuron):
    """Represents the Director neuron at the core of the cube."""

    def __init__(
        self,
        name,
        x,
        y,
        z,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
    ):
        super().__init__(
            name,
            x,
            y,
            z,
            "Director",
            opacity_threshold,
            emission_intensity,
            intensity_decay,
            reflectance,
            bias,
        )


class Communicator(Neuron):
    """Represents a Communicator neuron that connects different levels."""

    def __init__(
        self,
        name,
        x,
        y,
        z,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
    ):
        super().__init__(
            name,
            x,
            y,
            z,
            "Communicator",
            opacity_threshold,
            emission_intensity,
            intensity_decay,
            reflectance,
            bias,
        )


class Level1(Neuron):
    """Represents a neuron in Level 1 of the cube."""

    def __init__(
        self,
        name,
        x,
        y,
        z,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
    ):
        super().__init__(
            name,
            x,
            y,
            z,
            "Level1",
            opacity_threshold,
            emission_intensity,
            intensity_decay,
            reflectance,
            bias,
        )


class Level2(Neuron):
    """Represents a neuron in Level 2 of the cube."""

    def __init__(
        self,
        name,
        x,
        y,
        z,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
    ):
        super().__init__(
            name,
            x,
            y,
            z,
            "Level2",
            opacity_threshold,
            emission_intensity,
            intensity_decay,
            reflectance,
            bias,
        )


class Level3(Neuron):
    """Represents a neuron in Level 3 of the cube."""

    def __init__(
        self,
        name,
        x,
        y,
        z,
        opacity_threshold,
        emission_intensity,
        intensity_decay,
        reflectance,
        bias=0.0,
    ):
        super().__init__(
            name,
            x,
            y,
            z,
            "Level3",
            opacity_threshold,
            emission_intensity,
            intensity_decay,
            reflectance,
            bias,
        )


class MetaNeuron:
    """
    Represents a group of associated neurons acting as a single unit.

    Attributes:
        neurons (list): List of Neuron objects that are part of this association.
        precision (int): Total precision of the MetaNeuron in bits, calculated as the sum of precisions of its constituent neurons.
    """

    def __init__(self, neurons):
        self.neurons = neurons
        self.precision = sum([n.precision for n in neurons])
        # ... (Methods for calculations with increased precision will be added here) ...


# --- Batch Normalization ---
class BatchNorm:
    """
    Implements batch normalization to stabilize and accelerate training.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            self.running_mean = (
                self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * batch_var + (1 - self.momentum) * self.running_var
            )
            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta


# --- Quantization Function ---
def quantize(value, bits=BIT_PRECISION):
    """
    Quantizes a value to a given number of bits.
    """
    max_value = 2**bits - 1
    return round(value * max_value) / max_value


# --- Neural Cube Initialization ---
def initialize_neural_cube(grid):
    """
    Initializes the optical cube with neurons arranged in a 3D grid,
    following a Matryoshka doll structure (expanding cubes for each level).
    Also adds the neurons to the spatial grid.

    Args:
        grid (Grid): The spatial grid to add the neurons to.

    Returns:
        list: List of neurons in the cube.
    """
    neural_cube_data = []
    neuron_id = 1  # Unique ID for each neuron, starting with the Director (1)

    grid_size = 1  # Initial size of the grid (for level 0)
    for level in range(LEVELS_X + 1):
        # 3D Grid: Not actually used, could be removed.
        grid_np = np.zeros((grid_size, grid_size, grid_size), dtype=int)
        distance_between_neurons = 1 / grid_size  # Distance between neurons

        # Traverse the grid to place the neurons
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Avoid the center in higher levels (occupied by inner cubes)
                    if (
                        level == 0
                        or (i != grid_size // 2 or j != grid_size // 2)
                        or k != grid_size // 2
                    ):
                        x = (i + 0.5) * distance_between_neurons
                        y = (j + 0.5) * distance_between_neurons
                        z = (k + 0.5) * distance_between_neurons

                        # Neuron type assignment based on level and position
                        if neuron_id == 1:
                            neuron_type = "Director"
                        elif i == 0 and j == 0 and k == 0:
                            neuron_type = "Communicator"
                        else:
                            neuron_type = f"Level{level}"

                        # Create the neuron
                        neuron = globals()[neuron_type](
                            f"{neuron_type}-{neuron_id}",
                            x,
                            y,
                            z,
                            random.uniform(
                                MIN_OPACITY_THRESHOLD, MAX_OPACITY_THRESHOLD
                            ),
                            random.uniform(
                                MIN_EMISSION_INTENSITY, MAX_EMISSION_INTENSITY
                            ),
                            random.uniform(MIN_INTENSITY_DECAY, MAX_INTENSITY_DECAY),
                            random.uniform(MIN_REFLECTANCE, MAX_REFLECTANCE),
                            random.uniform(0.1, 0.3),
                        )
                        neural_cube_data.append(neuron)
                        grid.add_neuron(
                            neuron, len(neural_cube_data) - 1
                        )  # Add the neuron to the grid using its index
                        neuron_id += 1

        grid_size *= 2  # Expand the grid for the next level (like a Matryoshka doll)

    # Output neurons
    output_spacing = (
        1 / OUTPUT_NEURON_COUNT
    )  # Space between output neurons
    for i in range(OUTPUT_NEURON_COUNT):
        x = (i + 0.5) * output_spacing
        neuron = Neuron(
            f"Output-{i+1}",
            x,
            0,
            0,
            "Output",
            quantize(random.uniform(MIN_OPACITY_THRESHOLD, MAX_OPACITY_THRESHOLD)),
            quantize(random.uniform(MIN_EMISSION_INTENSITY, MAX_EMISSION_INTENSITY)),
            quantize(random.uniform(MIN_INTENSITY_DECAY, MAX_INTENSITY_DECAY)),
            quantize(random.uniform(MIN_REFLECTANCE, MAX_REFLECTANCE)),
            quantize(random.uniform(0.1, 0.3)),
        )
        neural_cube_data.append(neuron)
        grid.add_neuron(
            neuron, len(neural_cube_data) - 1
        )  # Add output neuron to the grid

    # --- Bias Neuron ---
    neural_cube_data.append(
        Neuron("Bias", 0, 0, 0, "Bias", 0, 0, 0, 1.0, 0)
    )  # Add bias neuron, intensity always = 1, not trainable
    grid.add_neuron(
        neural_cube_data[-1], len(neural_cube_data) - 1
    )  # Add bias neuron to the grid

    return neural_cube_data


# --- Neuron Group Definition ---
# Grouping is done dynamically based on neuron type, no need to hardcode indices
def get_neuron_groups(neural_cube_data):
    """
    Groups neurons based on their types. This is used for analysis and training.
    """
    return [
        [n for n in neural_cube_data if n.neuron_type == "Level1"],
        [n for n in neural_cube_data if n.neuron_type == "Level2"],
        [n for n in neural_cube_data if n.neuron_type == "Level3"],
        [n for n in neural_cube_data if n.neuron_type in ["Director", "Communicator"]],  # Group Director + Communicators
    ]


# --- Initialize batch normalization ---
# Assuming you want to normalize intensities for all trainable neurons
def initialize_batch_norm(neural_cube_data):
    """
    Initializes the batch normalization layer for the network.
    """
    trainable_neurons = [n for n in neural_cube_data if n.neuron_type != "Bias"]
    return BatchNorm(len(trainable_neurons))


# Initial reflectances
group_reflectances = [1.0, 1.0, 1.0, 1.0]

# --- Optical Path Matrix Calculation (CUDA) ---
@cuda.jit(device=True)
def calculate_distance_gpu(x1, y1, z1, x2, y2, z2):
    """Calculates the Euclidean distance between two points in 3D space (for GPU)."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_distance(x1, y1, z1, x2, y2, z2):
    """Calculates the Euclidean distance between two points in 3D space (for CPU)."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


@cuda.jit
def calculate_optical_path_matrix_kernel(neuron_count, neurons, optical_path_matrix):
    """
    CUDA kernel to calculate the optical path matrix, representing distances between neurons.
    """
    i, j = cuda.grid(2)
    if i < neuron_count and j < neuron_count:
        n1 = neurons[i]
        n2 = neurons[j]
        optical_path_matrix[i, j] = calculate_distance_gpu(
            n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]
        )


def calculate_optical_path_matrix(neurons):
    """
    Calculates the optical path matrix using GPU acceleration (CUDA).
    The matrix represents the distances between each pair of neurons in the cube.
    """
    neuron_count = len(neurons)

    # Initialize the matrix on the GPU using CuPy
    optical_path_matrix = cp.zeros((neuron_count, neuron_count), dtype=cp.float32)

    # Create the neuron array on the GPU using CuPy
    neurons_flat = cp.array([(n.x, n.y, n.z) for n in neurons], dtype=cp.float32)

    # Adjust grid and block dimensions for optimal GPU utilization
    threads_per_block = (
        16,
        16,
    )  # Experiment with higher values (multiples of 32)
    blocks_per_grid_x = (
        neuron_count + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blocks_per_grid_y = (
        neuron_count + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the CUDA kernel
    calculate_optical_path_matrix_kernel[blocks_per_grid, threads_per_block](
        neuron_count, neurons_flat, optical_path_matrix
    )

    # Synchronize to ensure calculations are finished and return the matrix
    cp.cuda.Stream.null.synchronize()
    return optical_path_matrix


# --- Light Propagation (CUDA) ---
@cuda.jit
def propagate_light_kernel(
    neural_cube_intensity,
    neuron_positions,
    reflectances,
    max_distance,
    grid_gpu,
    grid_shape,
    cell_size,
):
    """
    CUDA kernel for light propagation from each neuron to its neighbors using the spatial grid.
    """
    idx = cuda.grid(1)
    if idx < neuron_positions.shape[0]:
        x1, y1, z1 = neuron_positions[idx]
        cell_index_x = int(x1 // cell_size)
        cell_index_y = int(y1 // cell_size)
        cell_index_z = int(z1 // cell_size)

        # Iterate over neighboring cells
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    neighboring_cell_x = cell_index_x + i
                    neighboring_cell_y = cell_index_y + j
                    neighboring_cell_z = cell_index_z + k

                    # Check if the neighboring cell is within the grid bounds
                    if (
                        0 <= neighboring_cell_x < grid_shape[0]
                        and 0 <= neighboring_cell_y < grid_shape[1]
                        and 0 <= neighboring_cell_z < grid_shape[2]
                    ):
                        # Calculate the flattened index of the neighboring cell
                        cell_index = (
                            neighboring_cell_x * grid_shape[1] * grid_shape[2]
                            + neighboring_cell_y * grid_shape[2]
                            + neighboring_cell_z
                        )
                        neighbors = grid_gpu[cell_index]

                        # Iterate over neurons in the neighboring cell
                        for n in neighbors:
                            if n == -1:
                                break
                            if (
                                idx != n
                            ):  # Don't propagate light to itself
                                x2, y2, z2 = neuron_positions[n]
                                distance = (
                                    (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
                                ) ** 0.5
                                if distance < max_distance:
                                    intensity = reflectances[idx] / (distance**2)
                                    cuda.atomic.add(
                                        neural_cube_intensity, n, intensity
                                    )


def propagate_light_gpu(neurons, grid, max_distance):
    """
    Simulates light propagation from each neuron to its neighbors using the GPU and the spatial grid.

    Args:
        neurons (list): List of neurons.
        grid (Grid): The spatial grid containing the neurons.
        max_distance (float): Maximum light propagation distance.
    """
    neuron_positions = cp.array(
        [[n.x, n.y, n.z] for n in neurons], dtype=cp.float32
    )
    reflectances = cp.array([n.reflectance for n in neurons], dtype=cp.float32)
    neural_cube_intensity = cp.zeros(len(neurons), dtype=cp.float32)

    grid_shape = grid.dimensions // grid.cell_size

    # 1. Calcular el número máximo de vecinos por celda
    max_neighbors = max(
        [len(cell_neurons) for cell_neurons in grid.grid.values()]
    )

    # 2. Crear un array 2D en la GPU para almacenar los índices de neuronas
    grid_gpu = cp.full(
        (grid_shape[0] * grid_shape[1] * grid_shape[2], max_neighbors),
        -1,
        dtype=cp.int32,
    )

    # 3. Llenar grid_gpu con los índices de neuronas
    for cell_index, cell_neurons in grid.grid.items():
        flat_index = (
            cell_index[0] * grid_shape[1] * grid_shape[2]
            + cell_index[1] * grid_shape[2]
            + cell_index[2]
        )
        grid_gpu[flat_index, : len(cell_neurons)] = cp.array(
            cell_neurons, dtype=cp.int32
        )

    # 4. Ajustar las dimensiones del kernel
    threads_per_block = 256  # Ajuste basado en la arquitectura de la GPU
    blocks_per_grid = (
        len(neuron_positions) + threads_per_block - 1
    ) // threads_per_block

    # 5. Lanzar el kernel CUDA
    propagate_light_kernel[blocks_per_grid, threads_per_block](
        neural_cube_intensity,
        neuron_positions,
        reflectances,
        max_distance,
        grid_gpu,
        grid_shape,
        grid.cell_size,
    )

    # Update neuron intensities
    for i, neuron in enumerate(neurons):
        if neuron.neuron_type != "Bias":
            neuron.received_intensity += neural_cube_intensity[i].get()
            neuron.received_intensity = min(
                max(neuron.received_intensity, -10), 10
            )
            neuron.received_intensity = quantize(neuron.received_intensity)
            neuron.update_activation()

    all_neurons_in_groups = [
        n for group in get_neuron_groups(neurons) for n in group
    ]
    normalize_intensity(all_neurons_in_groups, eps=1e-8)


# --- OFDM Encoding and Decoding ---
def ofdm_modulate(data, carrier_count=CARRIER_COUNT):
    """
    Modulate data using OFDM.

    Args:
        data (np.ndarray): The data to modulate.
        carrier_count (int): Number of carriers.

    Returns:
        np.ndarray: The OFDM modulated signal.
    """
    symbols = np.fft.ifft(data, carrier_count)
    return np.concatenate([symbols[-carrier_count // 4 :], symbols])


def ofdm_demodulate(signal, carrier_count=CARRIER_COUNT):
    """
    Demodulate OFDM signal.

    Args:
        signal (np.ndarray): The OFDM signal to demodulate.
        carrier_count (int): Number of carriers.

    Returns:
        np.ndarray: The demodulated data.
    """
    symbols = signal[carrier_count // 4 : carrier_count // 4 + carrier_count]
    return np.fft.fft(symbols, carrier_count)


def generate_data_sequence(number, bit_length=CARRIER_COUNT):
    """
    Generates a binary sequence representing a number.

    Args:
        number (int): The number to encode.
        bit_length (int): The length of the bit sequence.

    Returns:
        np.ndarray: The binary sequence.
    """
    return np.array([int(bit) for bit in format(number, f"0{bit_length}b")], dtype=np.float64) 


def decode_data_sequence(sequence):
    """
    Decodes a binary sequence to retrieve the original number.

    Args:
        sequence (np.ndarray): The binary sequence.

    Returns:
        int: The decoded number.
    """
    return int("".join(map(str, sequence.astype(int))), 2)


def transmit_data(data, carrier_count=CARRIER_COUNT):
    """
    Transmit data using OFDM.

    Args:
        data (int): The data to transmit.
        carrier_count (int): Number of carriers.
    """
    data_sequence = generate_data_sequence(data, carrier_count)
    ofdm_signal = ofdm_modulate(data_sequence, carrier_count)
    # Transmit ofdm_signal to the neural cube (implementation depends on your system)
    return ofdm_signal


def receive_data(ofdm_signal, carrier_count=CARRIER_COUNT):
    """
    Receive data using OFDM.

    Args:
        ofdm_signal (np.ndarray): The OFDM signal received.
        carrier_count (int): Number of carriers.

    Returns:
        int: The received data.
    """
    demodulated_data = ofdm_demodulate(ofdm_signal, carrier_count)
    demodulated_sequence = (np.real(demodulated_data) > 0).astype(np.float32)
    return decode_data_sequence(demodulated_sequence)


# --- Neuron Association Functions ---


def find_neighbors(neuron, neural_cube_data, max_distance=1):
    """
    Finds neighboring neurons within a specified maximum distance.
    """
    neighbors = []
    for other_neuron in neural_cube_data:
        if other_neuron != neuron and calculate_distance(
            neuron.x,
            neuron.y,
            neuron.z,
            other_neuron.x,
            other_neuron.y,
            other_neuron.z,
        ) <= max_distance:
            neighbors.append(other_neuron)
    return neighbors


def calculate_entropy(neurons, parameter="reflectance"):
    """
    Calculates the Shannon entropy of a given parameter within a group of neurons.
    Entropy is a measure of information or uncertainty. Higher entropy implies more variability or information content.
    """
    values = [getattr(n, parameter) for n in neurons]
    probabilities = (
        np.histogram(values, bins=2 ** max([n.precision for n in neurons]))[0]
        / len(neurons)
    )
    return entropy(probabilities)


def evaluate_association(
    neuron, neighbors, target_precision, current_accuracy, cost_function
):
    """
    Evaluates the potential benefit of neuron association.
    It checks different group sizes of neighboring neurons to find the group that
    provides the maximum benefit (improvement in accuracy minus the cost of association).
    """
    best_group = []
    best_benefit = 0

    for group_size in range(1, len(neighbors) + 1):
        group = neighbors[:group_size]
        group_precision = sum([n.precision for n in group])

        if group_precision >= target_precision:
            association_cost = cost_function(group)
            potential_accuracy = estimate_accuracy_improvement(
                group, current_accuracy
            )
            benefit = potential_accuracy - association_cost

            if benefit > best_benefit:
                best_benefit = benefit
                best_group = group

    return best_group


def form_group(neuron, neighbors):
    """
    Forms a MetaNeuron by associating a neuron with its neighbors.
    """
    meta_neuron = MetaNeuron([neuron] + neighbors)
    neuron.meta_neuron = meta_neuron
    for n in neighbors:
        n.meta_neuron = meta_neuron
        n.associated_neurons = neighbors


def dissolve_group(neuron):
    """
    Dissolves a neuron association (MetaNeuron).
    """
    for n in neuron.meta_neuron.neurons:
        n.meta_neuron = None
        n.associated_neurons = []


# --- Placeholder functions ---
def estimate_accuracy_improvement(group, current_accuracy):
    """
    Estimates how much the accuracy will improve if a group of neurons is formed.
    This is a placeholder function that needs to be implemented based on your model's specifics.
    """
    # Replace this with your logic to estimate accuracy improvement
    return current_accuracy * 0.1


def association_cost_function(group):
    """
    Calculates the cost of associating a group of neurons.
    This is a placeholder function that needs to be implemented based on factors like
    communication overhead and synchronization time.
    """
    # Replace this with your logic to calculate the association cost
    return len(group) * 0.01


# --- Group Processing Functions ---
def process_group(group, data):
    """Processes data for a group of neurons."""
    neuron_outputs = [process_data(neuron, data) for neuron in group]
    return combine_outputs(neuron_outputs)


def combine_outputs(outputs):
    """Combines the outputs of individual neurons in a group."""
    combined_output = 0
    for output in outputs:
        combined_output = (combined_output << 2) | output
    return combined_output


# --- Placeholder for individual neuron data processing ---
def process_data(neuron, data):
    """Placeholder for processing data at the individual neuron level."""
    return int(neuron.received_intensity * 2)


# --- Central Neuron Processing ---
def process_central_outputs(group_outputs, neural_cube_data, group_reflectances):
    """Processes outputs from all neuron groups using the Director neuron."""
    central_neuron = neural_cube_data[
        0
    ]  # Assuming the first neuron is the Director neuron
    combined_output = 0
    for group_output, reflectance in zip(group_outputs, group_reflectances):
        combined_output += group_output * reflectance
    group_performance = evaluate_performance(combined_output)
    for i, group in enumerate(get_neuron_groups(neural_cube_data)):
        if i < len(group_performance):
            if group_performance[i] > 0:
                reward_group(central_neuron, i, group_performance[i])
            else:
                penalize_group(central_neuron, i, abs(group_performance[i]))


# --- Placeholder functions (need implementation) ---
def encode_output(combined_output):
    """Placeholder: Encodes the output."""
    return transmit_data(combined_output)  # Placeholder


def send_output(encoded_sequence):
    """Placeholder: Sends the output sequence."""
    pass
    # print("Sequence sent:", pam_sequence)  # Placeholder: Print the sequence


def evaluate_performance(combined_output):
    """Placeholder: Evaluates the performance of the cube."""
    return [1, -1, 0, 1]  # Placeholder for performance evaluation


# --- Reward and Penalty Functions ---
def reward_group(central_neuron, group_index, reward):
    """Rewards a neuron group by increasing its reflectance."""
    group_reflectances[group_index] += reward * REWARD_RATE
    central_neuron.adjust_reflectance(reward)


def penalize_group(central_neuron, group_index, penalty):
    """Penalizes a neuron group by decreasing its reflectance."""
    group_reflectances[group_index] -= penalty * PENALTY_RATE
    central_neuron.adjust_reflectance(-penalty)


# --- Function to Adjust a Single Parameter ---
def adjust_parameter(param, step_size):
    """Adjusts a parameter by a given step size."""
    return max(param - step_size, 0) if param > 0 else min(param + step_size, 1)


# --- Fine-Tuning Function (Ray-compatible) ---
@ray.remote(num_gpus=1)
def fine_tune_parameters_ray(iterations, neural_cube_data, optical_path_matrix):
    """
    Fine-tunes global parameters (RECURRENCE_FACTOR, INTENSITY_DECAY, etc.)
    using the DEAP evolutionary algorithm to optimize accuracy.
    """
    global RECURRENCE_FACTOR, INTENSITY_DECAY, REFLECTANCE_DECAY, HEBBIAN_LEARNING_RATE, REWARD_RATE, PENALTY_RATE
    best_accuracy = 0.0
    optical_path_matrix_gpu = cuda.to_device(optical_path_matrix)
    for iteration in range(iterations):
        print(f"Fine-tuning iteration {iteration + 1}...")
        current_accuracy = evaluate_parameters(
            neural_cube_data, optical_path_matrix_gpu
        )
        print(f"Current accuracy: {current_accuracy:.4f}")
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
        else:
            RECURRENCE_FACTOR = adjust_parameter(
                RECURRENCE_FACTOR, FINE_TUNE_STEP_SIZE
            )
            INTENSITY_DECAY = adjust_parameter(
                INTENSITY_DECAY, FINE_TUNE_STEP_SIZE
            )
            REFLECTANCE_DECAY = adjust_parameter(
                REFLECTANCE_DECAY, FINE_TUNE_STEP_SIZE
            )
            HEBBIAN_LEARNING_RATE = adjust_parameter(
                HEBBIAN_LEARNING_RATE, FINE_TUNE_STEP_SIZE
            )
            REWARD_RATE = adjust_parameter(REWARD_RATE, FINE_TUNE_STEP_SIZE)
            PENALTY_RATE = adjust_parameter(PENALTY_RATE, FINE_TUNE_STEP_SIZE)
    return best_accuracy


def evaluate_parameters(neural_cube_data, optical_path_matrix_gpu):
    """
    Evaluates the performance of the cube with the current global parameters.
    Used by the fine-tuning function.
    """
    total_correct = 0
    total_incorrect = 0
    for _ in range(FINE_TUNE_FLASHES):
        num1 = random.randint(0, 2**BIT_PRECISION - 1)
        num2 = random.randint(0, 2**BIT_PRECISION - 1)
        
        ofdm_signal1 = transmit_data(num1)
        ofdm_signal2 = transmit_data(num2)
        group_outputs = [
            process_group(group, [ofdm_signal1, ofdm_signal2])
            for group in get_neuron_groups(neural_cube_data)
        ]
        process_central_outputs(
            group_outputs, neural_cube_data, group_reflectances
        )
        for pulse in range(NUM_PULSES):
            for neuron in neural_cube_data:
                neuron.received_intensity = 0.0
            time.sleep(FLASH_INTERVAL)
            for i, neuron in enumerate(neural_cube_data[:-OUTPUT_NEURON_COUNT]):
                propagate_light_gpu(
                    neural_cube_data, grid, MAX_RAY_DISTANCE
                )  # Use the GPU-accelerated function with the grid
            all_neurons_in_groups = [
                n for group in get_neuron_groups(neural_cube_data) for n in group
            ]
            intensities = np.array(
                [n.received_intensity for n in all_neurons_in_groups]
            )
            normalized_intensities = batch_norm.forward(intensities)
            for n, p in zip(all_neurons_in_groups, normalized_intensities):
                n.received_intensity = p
            decoded_output = decode_output(neural_cube_data)
        for neuron in neural_cube_data:
            neuron.apply_reflectance_decay()
        target_output = num1 + num2
        if decoded_output == target_output:
            total_correct += 1
        else:
            total_incorrect += 1
    return total_correct / (total_correct + total_incorrect)


# --- Function to save model state ---
def save_model_state(neural_cube_data, filename="model_state.npy"):
    """
    Saves the current state of the model (neuron parameters, global parameters, etc.)
    to a file for later loading.
    """
    state = {
        "global_parameters": np.array(
            [
                RECURRENCE_FACTOR,
                INTENSITY_DECAY,
                REFLECTANCE_DECAY,
                HEBBIAN_LEARNING_RATE,
                REWARD_RATE,
                PENALTY_RATE,
            ]
        ),
        "neurons": {},
        "batch_norm": batch_norm,
        "group_reflectances": group_reflectances,
    }
    for neuron in neural_cube_data:
        state["neurons"][neuron.name] = {
            "type": neuron.neuron_type,
            "x": neuron.x,
            "y": neuron.y,
            "z": neuron.z,
            "reflectance": neuron.reflectance,
            "received_intensity": neuron.received_intensity,
            "opacity_threshold": neuron.opacity_threshold,
            "bias": neuron.bias,
            "learning_rate": neuron.learning_rate,
            "reflectance_decay": neuron.reflectance_decay,
            # ... other attributes ...
        }
    np.save(filename, state)


# --- Function to load model state ---
def load_model_state(filename="model_state.npy"):
    """
    Loads the model state from a previously saved file.
    """
    global RECURRENCE_FACTOR, INTENSITY_DECAY, REFLECTANCE_DECAY, HEBBIAN_LEARNING_RATE, REWARD_RATE, PENALTY_RATE, batch_norm, group_reflectances, grid
    neural_cube_data = initialize_neural_cube(grid)  # Pass the grid object to initialize_neural_cube
    if os.path.exists(filename):
        try:
            state = np.load(filename, allow_pickle=True).item()
            (
                RECURRENCE_FACTOR,
                INTENSITY_DECAY,
                REFLECTANCE_DECAY,
                HEBBIAN_LEARNING_RATE,
                REWARD_RATE,
                PENALTY_RATE,
            ) = state["global_parameters"]
            for neuron in neural_cube_data:
                # Find neuron data by name, assuming names are unique and consistent
                neuron_data = state["neurons"].get(neuron.name)
                if neuron_data:
                    neuron.reflectance = neuron_data["reflectance"]
                    neuron.received_intensity = neuron_data[
                        "received_intensity"
                    ]
                    neuron.opacity_threshold = neuron_data[
                        "opacity_threshold"
                    ]
                    neuron.bias = neuron_data["bias"]
                    neuron.learning_rate = neuron_data["learning_rate"]
                    neuron.reflectance_decay = neuron_data[
                        "reflectance_decay"
                    ]
                    # ... load other attributes ...
            batch_norm = state.get(
                "batch_norm", initialize_batch_norm(neural_cube_data)
            )  # load batch_norm state, if it exists
            group_reflectances = state.get(
                "group_reflectances", [1.0, 1.0, 1.0, 1.0]
            )  # Load group reflectances, if they exist
            print(f"Model state loaded from '{filename}'.")
        except Exception as e:
            print(f"Error loading model state from '{filename}': {e}")
    else:
        print(
            f"Model state file '{filename}' not found. Using default values."
        )
    return neural_cube_data


# --- Watchdog Functions ---
def monitor_activity(neural_cube_data):
    """Monitors the percentage of active neurons in the cube."""
    active_neuron_count = sum(
        [1 for neuron in neural_cube_data if neuron.active]
    )
    return active_neuron_count / len(neural_cube_data) * 100


def monitor_intensity(neural_cube_data):
    """
    Monitors intensity statistics: mean, standard deviation, saturated neurons, and inactive neurons.
    """
    intensities = np.array(
        [neuron.received_intensity for neuron in neural_cube_data]
    )
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)
    saturated_neurons = np.sum(intensities >= 3) / len(intensities) * 100
    inactive_neurons = np.sum(intensities == 0) / len(intensities) * 100
    return {
        "mean": mean_intensity,
        "std": std_intensity,
        "saturated": saturated_neurons,
        "inactive": inactive_neurons,
    }


def monitor_decoder_output(
    decoder, neural_cube_data, test_data_size=100
):
    """
    Monitors the frequency of different decoded outputs from the cube.
    Used to detect patterns and potential issues in the decoding process.
    """
    output_counts = {}
    for _ in range(test_data_size):
        random_number = random.randint(0, 2**BIT_PRECISION - 1)
        decoded_output = decoder(neural_cube_data)
        output_counts[decoded_output] = (
            output_counts.get(decoded_output, 0) + 1
        )
    return output_counts


def monitor_global_parameters():
    """Returns a dictionary of the current global learning parameters."""
    return {
        "RECURRENCE_FACTOR": RECURRENCE_FACTOR,
        "INTENSITY_DECAY": INTENSITY_DECAY,
        "REFLECTANCE_DECAY": REFLECTANCE_DECAY,
        "HEBBIAN_LEARNING_RATE": HEBBIAN_LEARNING_RATE,
        "REWARD_RATE": REWARD_RATE,
        "PENALTY_RATE": PENALTY_RATE,
    }


def analyze_deviations(
    activity_percentage,
    intensity_stats,
    global_parameters,
    activity_threshold=(50, 95),
):
    """
    Analyzes deviations in neuron activity, intensity, and global parameters.
    Helps the watchdog identify potential issues in the cube's operation.
    """
    deviations = []
    if not activity_threshold[0] <= activity_percentage <= activity_threshold[
        1
    ]:
        deviations.append(
            f"Activity out of bounds: {activity_percentage:.2f}% (expected {activity_threshold[0]}-{activity_threshold[1]}%)"
        )
    if intensity_stats["saturated"] > 20:
        deviations.append(
            f"High neuron saturation: {intensity_stats['saturated']:.2f}%"
        )
    if intensity_stats["inactive"] > 20:
        deviations.append(
            f"High neuron inactivity: {intensity_stats['inactive']:.2f}%"
        )
    if global_parameters["RECURRENCE_FACTOR"] < 0.1:
        deviations.append(
            f"Low RECURRENCE_FACTOR: {global_parameters['RECURRENCE_FACTOR']:.2f}"
        )
    return deviations


def analyze_patterns(decoder_output_frequency):
    """
    Analyzes patterns in the decoder output frequency to detect potential issues.
    For example, low entropy might indicate a lack of diversity in outputs.
    """
    probabilities = np.array(
        list(decoder_output_frequency.values())
    ) / sum(decoder_output_frequency.values())
    entropy_value = entropy(probabilities)
    patterns = []
    if entropy_value < 1.0:
        patterns.append(f"Low output entropy: {entropy_value:.2f}")
    for output, count in decoder_output_frequency.items():
        if count / sum(decoder_output_frequency.values()) > 0.5:
            patterns.append(f"Dominant output: {output} ({count} occurrences)")
    return patterns


def adjust_thresholds(
    neural_cube_data,
    activity_percentage,
    activity_threshold=(50, 95),
    learning_rate=0.01,
):
    """
    Adjusts neuron opacity thresholds to maintain a desired level of activity in the cube.
    """
    target_activity = np.mean(activity_threshold)
    error = activity_percentage - target_activity
    for neuron in neural_cube_data:
        neuron.opacity_threshold -= learning_rate * error


def rebalance_reflectances(group_reflectances):
    """Rebalances reflectances of neuron groups to ensure a fair contribution from each group."""
    total_reflectance = sum(group_reflectances)
    for i in range(len(group_reflectances)):
        group_reflectances[i] /= total_reflectance


def adjust_decoder(decoder, intensity_stats, margin=0.1):
    """Placeholder: Adjusts the decoder based on intensity statistics."""
    pass  # Placeholder (needs implementation)


# --- Divergence Function (Mean Squared Error) ---
def mean_squared_error(outputs, targets):
    """Calculates the mean squared error between the outputs and target values."""
    return np.mean(np.square(outputs - targets))


# --- Calculation of the Intensity Gradient ---
def calculate_gradient(neurons, target_output):
    """
    Calculates the gradient of the divergence function with respect to neuron reflectances.
    The gradient is used to update reflectances during training.
    """
    outputs = np.array(
        [decode_output(neurons) for _ in range(len(neurons))]
    )  # Decode output for each neuron
    return (
        2 * (outputs - target_output) / len(neurons),
        mean_squared_error(outputs, target_output),
    )


# --- Update Reflectances Based on Gradient ---
def update_reflectances(neurons, gradients, learning_rate=HEBBIAN_LEARNING_RATE):
    """
    Updates neuron reflectances based on the calculated gradient and the Hebbian learning rate.
    """
    for neuron, gradient in zip(neurons, gradients):
        neuron.reflectance -= learning_rate * gradient
        neuron.reflectance = quantize(max(0, min(neuron.reflectance, 2)))


# --- Gradient Clipping ---
def clip_gradient(gradient, max_value=1.0):
    """Clips the gradient to prevent exploding gradients."""
    return np.clip(gradient, -max_value, max_value)


# --- Intensity Normalization Function ---
def normalize_intensity(neurons, eps=1e-8):
    """
    Normalizes neuron intensities to have zero mean and unit variance.
    This helps improve training stability and convergence.
    """
    intensities = np.array([n.received_intensity for n in neurons])
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)
    for neuron in neurons:
        neuron.received_intensity = (
            neuron.received_intensity - mean_intensity
        ) / (std_intensity + eps)


# --- Network Activity Manager ---
class NetworkActivityManager:
    """
    Manages the overall activity level of the network.
    It can increase or decrease neuron activity based on predefined thresholds.
    """

    def __init__(self, neural_cube_data, activity_threshold=(0.2, 0.8)):
        self.neural_cube_data = neural_cube_data
        self.activity_threshold = activity_threshold

    def monitor_activity(self):
        """Returns the percentage of active neurons in the cube."""
        return sum(
            [1 for neuron in self.neural_cube_data if neuron.active]
        ) / len(self.neural_cube_data)

    def adjust_activity(self):
        """Adjusts neuron thresholds to regulate activity level."""
        activity_level = self.monitor_activity()
        if activity_level < self.activity_threshold[0]:
            self.increase_activity()
        elif activity_level > self.activity_threshold[1]:
            self.decrease_activity()

    def increase_activity(self):
        """Increases overall neuron activity by reducing opacity thresholds and increasing reflectances."""
        for neuron in self.neural_cube_data:
            neuron.opacity_threshold *= 0.9
            neuron.reflectance *= 1.1

    def decrease_activity(self):
        """Decreases overall neuron activity by increasing opacity thresholds and reducing reflectances."""
        for neuron in self.neural_cube_data:
            neuron.opacity_threshold *= 1.1
            neuron.reflectance *= 0.9


def watchdog(
    neural_cube_data,
    neuron_groups,
    group_reflectances,
    ofdm_decoder,
    intensity=WATCHDOG_INTENSITY,
    check_duration=WATCHDOG_CHECK_DURATION,
    rest_duration=WATCHDOG_REST_DURATION,
    optical_path_matrix=None,
):
    """
    Acts as a watchdog to monitor the health and performance of the cube.
    It checks for deviations in activity, intensity, and parameters, and can take corrective actions.
    """
    if intensity == 0:
        return  # Watchdog disabled
    activity_manager = NetworkActivityManager(neural_cube_data)
    while True:
        start_time = time.time()
        activity_percentage = monitor_activity(neural_cube_data)
        intensity_stats = monitor_intensity(neural_cube_data)
        decoder_output_frequency = monitor_decoder_output(
            ofdm_decoder, neural_cube_data
        )
        global_parameters = monitor_global_parameters()
        deviations = analyze_deviations(
            activity_percentage, intensity_stats, global_parameters
        )
        patterns = analyze_patterns(decoder_output_frequency)
        if deviations:
            adjust_thresholds(neural_cube_data, activity_percentage)
            rebalance_reflectances(group_reflectances)
            best_params = adjust_global_parameters(
                list(global_parameters.values()),
                deviations,
                neural_cube_data,
                optical_path_matrix,
            )
            (
                RECURRENCE_FACTOR,
                INTENSITY_DECAY,
                REFLECTANCE_DECAY,
                HEBBIAN_LEARNING_RATE,
                REWARD_RATE,
                PENALTY_RATE,
            ) = best_params
        if patterns:
            adjust_decoder(ofdm_decoder, intensity_stats)
        activity_manager.adjust_activity()
        print("Watchdog actions:")
        if deviations:
            print("- Deviations detected:", deviations)
        if patterns:
            print("- Patterns detected:", patterns)
        elapsed_time = time.time() - start_time
        actual_rest_duration = rest_duration / intensity
        time.sleep(max(0, actual_rest_duration - elapsed_time))


# --- Evaluation Function for Network Performance (Used by DEAP) ---
def evaluate_network(parameters, neural_cube_data, optical_path_matrix):
    """
    Evaluates the performance of the optical cube using the given parameters.
    This function is used by the DEAP evolutionary algorithm to find the best set of global parameters.
    """
    global RECURRENCE_FACTOR, INTENSITY_DECAY, REFLECTANCE_DECAY, HEBBIAN_LEARNING_RATE, REWARD_RATE, PENALTY_RATE
    (
        RECURRENCE_FACTOR,
        INTENSITY_DECAY,
        REFLECTANCE_DECAY,
        HEBBIAN_LEARNING_RATE,
        REWARD_RATE,
        PENALTY_RATE,
    ) = parameters
    total_correct = 0
    total_incorrect = 0
    for _ in range(FINE_TUNE_FLASHES):
        num1 = random.randint(0, 2**BIT_PRECISION - 1)
        num2 = random.randint(0, 2**BIT_PRECISION - 1)
        ofdm_signal1 = transmit_data(num1)
        ofdm_signal2 = transmit_data(num2)
        group_outputs = [
            process_group(group, [ofdm_signal1, ofdm_signal2])
            for group in get_neuron_groups(neural_cube_data)
        ]
        process_central_outputs(
            group_outputs, neural_cube_data, group_reflectances
        )
        for pulse in range(NUM_PULSES):
            for neuron in neural_cube_data:
                neuron.received_intensity = 0.0
            time.sleep(FLASH_INTERVAL)
            for i, neuron in enumerate(neural_cube_data[:-OUTPUT_NEURON_COUNT]):
                propagate_light_gpu(
                    neural_cube_data, grid, MAX_RAY_DISTANCE
                )  # Use the GPU-accelerated function with the grid
            all_neurons_in_groups = [
                n for group in get_neuron_groups(neural_cube_data) for n in group
            ]
            intensities = np.array(
                [n.received_intensity for n in all_neurons_in_groups]
            )
            normalized_intensities = batch_norm.forward(intensities)
            for n, p in zip(all_neurons_in_groups, normalized_intensities):
                n.received_intensity = p
            decoded_output = decode_output(neural_cube_data)
        for neuron in neural_cube_data:
            neuron.apply_reflectance_decay()
        target_output = num1 + num2
        if decoded_output == target_output:
            total_correct += 1
        else:
            total_incorrect += 1
    return total_correct / (total_correct + total_incorrect)


# --- Function to Adjust Global Parameters (Using DEAP) ---
def adjust_global_parameters(
    global_parameters, deviations, neural_cube_data, optical_path_matrix
):
    """
    Uses the DEAP evolutionary algorithm to find the best set of global parameters
    that address the identified deviations in the cube's performance.
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=len(global_parameters),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        evaluate_network,
        neural_cube_data=neural_cube_data,
        optical_path_matrix=optical_path_matrix,
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    (
        population,
        logbook,
    ) = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )  # Turn off verbose output from DEAP
    return hof[0]

def decode_output(neurons):  # Define decode_output globally 
    """Decodes the output from the output neurons."""
    output_neurons = [n for n in neurons if n.neuron_type == "Output"]
    result = 0
    for i, neuron in enumerate(output_neurons):
        if neuron.active:
            result += 2**i
    return result


# --- Main Execution ---
if __name__ == "__main__":
    # Create the spatial grid
    grid = Grid(GRID_DIMENSIONS, CELL_SIZE)
    neural_cube_data = load_model_state()  # Load the model state
    iteration = 0  # Iteration counter

    print("Initializing 3D Optical Cube...")
    optical_path_matrix = calculate_optical_path_matrix(neural_cube_data)
    print("Optical path matrix calculated.")

    neuron_groups = get_neuron_groups(neural_cube_data)
    batch_norm = initialize_batch_norm(neural_cube_data)

    try:
        num_gpus_available = len(GPUtil.getGPUs())
        ray.init(num_gpus=num_gpus_available)
        print(f"Ray initialized with {num_gpus_available} GPUs.")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        ray.init()  # Initialize Ray in single-process mode
        print("Ray initialized in single-process mode.")

    if AUTO_FINE_TUNING_X > 0:
        print(
            f"Fine-tuning parameters with {AUTO_FINE_TUNING_X} processes..."
        )
        results = ray.get(
            [
                fine_tune_parameters_ray.remote(
                    FINE_TUNE_ITERATIONS // AUTO_FINE_TUNING_X,
                    neural_cube_data,
                    optical_path_matrix,
                )
                for _ in range(AUTO_FINE_TUNING_X)
            ]
        )
        best_accuracy = max(results)
        print(f"Fine-tuning complete. Best accuracy: {best_accuracy:.4f}")
    else:
        print("Fine-tuning disabled.")

    # Start watchdog
    if WATCHDOG_INTENSITY > 0:
        watchdog_thread = threading.Thread(
            target=watchdog,
            args=(
                neural_cube_data,
                neuron_groups,
                group_reflectances,
                decode_output,
                WATCHDOG_INTENSITY,
                WATCHDOG_CHECK_DURATION,
                WATCHDOG_REST_DURATION,
                optical_path_matrix,
            ),
        )
        watchdog_thread.daemon = True
        watchdog_thread.start()

    # Lists for training data
    accuracy_history = []
    divergence_history = []
    cpu_usage_history = []
    gpu_usage_history = []
    memory_usage_history = []
    start_time = time.time()

    # --- Training Loop (After Fine-Tuning) ---
    total_correct = 0
    total_incorrect = 0
    accuracy = 0.0  # Initialize accuracy here in the global scope

    # Parallel Training Logic
    if TRAINING_FLASH_X == 1:
        # Traditional Training: All neurons train together
        @ray.remote(num_gpus=1)
        def process_flash(flash_index, neural_cube_data, grid, current_accuracy):  
            """Processes a single training flash."""
            global group_reflectances, batch_norm, iteration
            decoded_output = decode_output(neural_cube_data)  # Call the global decode_output function
            num1 = random.randint(0, 2**BIT_PRECISION - 1)
            num2 = random.randint(0, 2**BIT_PRECISION - 1)
            ofdm_signal1 = transmit_data(num1)
            ofdm_signal2 = transmit_data(num2)
            group_outputs = [
                process_group(group, [ofdm_signal1, ofdm_signal2])
                for group in get_neuron_groups(neural_cube_data)
            ]
            process_central_outputs(
                group_outputs, neural_cube_data, group_reflectances
            )
            for pulse in range(min(NUM_PULSES, BIT_PRECISION)):
                for neuron in neural_cube_data:
                    neuron.received_intensity = 0.0
                time.sleep(FLASH_INTERVAL)
                for i, neuron in enumerate(
                    neural_cube_data[:-OUTPUT_NEURON_COUNT]
                ):
                    # propagate_light(neuron, neural_cube_data, optical_path_matrix, NEURON_COUNT, MAX_RAY_DISTANCE, input_sequence1[pulse])
                    # propagate_light(neuron, neural_cube_data, optical_path_matrix, NEURON_COUNT, MAX_RAY_DISTANCE, input_sequence2[pulse])
                    propagate_light_gpu(
                        neural_cube_data, grid, MAX_RAY_DISTANCE
                    )  # Use the GPU-accelerated function with the grid
                # Apply batch normalization
                trainable_neurons = [
                    n for n in neural_cube_data if n.neuron_type != "Bias"
                ]
                intensities = np.array(
                    [n.received_intensity for n in trainable_neurons]
                )
                normalized_intensities = batch_norm.forward(intensities)
                for n, p in zip(
                    trainable_neurons, normalized_intensities
                ):
                    n.received_intensity = p
                decoded_output = decode_output(
                    neural_cube_data
                )  # Decode the output

                # --- Neuron Association ---
                if iteration % ASSOCIATION_EVALUATION_INTERVAL == 0:
                    for neuron in neural_cube_data:
                        if neuron.neuron_type != "Bias":
                            neighbors = find_neighbors(
                                neuron, neural_cube_data
                            )
                            entropy_value = calculate_entropy(neighbors)

                            if neuron.meta_neuron is None:
                                best_group = evaluate_association(
                                    neuron,
                                    neighbors,
                                    TARGET_PRECISION,
                                    current_accuracy,
                                    association_cost_function,
                                )  # Use current_accuracy here
                                if best_group:
                                    form_group(neuron, best_group)
                            else:
                                if entropy_value < ENTROPY_THRESHOLD:
                                    dissolve_group(neuron)

            # Print debug info only once per flash
            print(f"Flash: {flash_index+1}")
            for i, neuron in enumerate(neural_cube_data):
                print(
                    f"  Neuron {i}: Active: {neuron.active}, Intensity: {neuron.received_intensity:.2f}"
                )
            for neuron in neural_cube_data:
                if (
                    neuron.neuron_type != "Bias"
                ):  # Don't apply reflectance decay to bias neuron
                    neuron.apply_reflectance_decay()
            target_output = num1 + num2
            if decoded_output == target_output:
                return 1  # Correct
            else:
                return 0  # Incorrect

        # Execute Traditional Training
        flash_results = [
            process_flash.remote(
                flash_index, neural_cube_data, grid, accuracy
            )
            for flash_index in range(FLASH_COUNT)
        ]  # Pass accuracy and grid here
        for flash_index in range(FLASH_COUNT):
            result = ray.get(flash_results[flash_index])
            if result == 1:
                total_correct += 1
            else:
                total_incorrect += 1

            iteration += 1

            # Update accuracy within the loop to ensure the latest value is passed
            if total_correct + total_incorrect > 0:
                accuracy = total_correct / (total_correct + total_incorrect)

    elif TRAINING_FLASH_X == 2:
        # Two parallel training groups
        @ray.remote(num_gpus=0.5)
        def process_flash_group1(flash_index, optical_path_matrix):
            # Your code to train the first group of neurons
            pass  # Placeholder for group 1 training logic

        @ray.remote(num_gpus=0.5)
        def process_flash_group2(flash_index, optical_path_matrix):
            # Your code to train the second group of neurons
            pass  # Placeholder for group 2 training logic

        flash_results_group1 = [
            process_flash_group1.remote(flash_index, optical_path_matrix)
            for flash_index in range(FLASH_COUNT)
        ]
        flash_results_group2 = [
            process_flash_group2.remote(flash_index, optical_path_matrix)
            for flash_index in range(FLASH_COUNT)
        ]
        for flash_index in range(FLASH_COUNT):
            result_group1 = ray.get(flash_results_group1[flash_index])
            result_group2 = ray.get(flash_results_group2[flash_index])
            # Your code to process the results from both groups

    # ... (Similar logic for TRAINING_FLASH_X == 3 and 4)

    # Initialize and run the visualizer if enabled
    if Neuron_Activity == 1:
        from Neuron_Activity import NeuronVisualizer

        visualizer = NeuronVisualizer(neural_cube_data)
        visualizer.run()

    # Calculate and store accuracy
    if total_correct + total_incorrect > 0:
        accuracy = total_correct / (total_correct + total_incorrect)
    else:
        accuracy = 0.0
    accuracy_history.append(accuracy)

    # Collect other data for reporting
    cpu_usage_history.append(psutil.cpu_percent())
    memory_usage_history.append(psutil.virtual_memory().percent)
    try:
        gpus = GPUtil.getGPUs()
        gpu_load = gpus[0].load * 100  # Get the load of the first available GPU
        gpu_usage_history.append(gpu_load)
    except Exception as e:
        print(f"Error getting GPU information: {e}")
        gpu_usage_history.append(0)  # Append 0 if GPU information is not available

    print(
        f"Final Score - Correct: {total_correct}, Incorrect: {total_incorrect}"
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    save_model_state(neural_cube_data)  # Save the model state

    # Generate a report if CUBE_REPORT is available
    if "CUBE_REPORT" in globals():
        CUBE_REPORT.generate_report(
            accuracy_history,
            divergence_history,
            cpu_usage_history,
            gpu_usage_history,
            memory_usage_history,
            neuron_groups,
            group_reflectances,
            monitor_global_parameters(),
            optical_path_matrix,
            decode_output, 
            neural_cube_data,   
        )
    else:
        print(
            "Warning: CUBE_REPORT module not found. Skipping report generation."
        )
