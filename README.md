# Terminator: Full Context Interaction with Hyper-Kernels

This repository contains the implementation of the Terminator architecture, which achieves full context interaction using large implicit kernels. The architecture combines a slow network, based on coordinate-based implicit MLPs, with a fast convolutional network. The HyperZ·Z·W operator connects hyper-kernels and hidden activations, allowing for context-varying weights and enhancing feature extraction. The architecture also includes a bottleneck layer for compressing concatenated channels and improving training convergence. Our model exhibits excellent properties, such as stable zero-mean features and faster training convergence, while requiring fewer model parameters. Experimental results on pixel-level 1D and 2D image classification benchmarks demonstrate the superior performance of our architecture.

## Features

- Full Context Interaction: The Terminator architecture achieves full context interaction using large implicit kernels and the HyperZ·Z·W operator.
- Multi-Branch Hidden Representations: Hyper-kernels of different sizes are integrated to produce multi-branch hidden representations, enhancing feature extraction.
- Bottleneck Layer: A bottleneck layer compresses concatenated channels, allowing only valuable information to propagate to subsequent layers.
- Stable Zero-Mean Features: Our model incorporates innovative components that result in stable zero-mean features.
- Faster Training Convergence: The architecture enables faster training convergence compared to traditional approaches.
- Fewer Model Parameters: Our model requires fewer model parameters, making it more efficient.

## Installation

To use the Terminator architecture, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Import the architecture into your project by adding `from terminator import Terminator` to your code.

## Usage

To use the Terminator architecture, follow these steps:

1. Initialize the architecture by creating an instance of the `Terminator` class.
2. Connect your slow and fast networks to the architecture using the `connect` method.
3. Start the architecture by calling the `start` method.
4. Use the architecture's API to exchange information and collaborate between the networks.

```python
from terminator import Terminator

# Initialize the architecture
terminator = Terminator()

# Connect slow and fast networks
terminator.connect(slow_network, fast_network)

# Start the architecture
terminator.start()

# Use the architecture's API to exchange information and collaborate
terminator.send_data(slow_network_data)
terminator.receive_data(fast_network_data)
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

