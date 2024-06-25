import torch
from terminator import Terminator

def main():
    # Example usage
    model = Terminator(num_classes=10)
    x = torch.randn(64, 3, 32, 32)
    output = model(x)
    print(output.shape)  # Should be (1, 10)

    # During training, you would add the slow neural loss to your main loss
    # loss = criterion(output, target) + lambda * model.slow_neural_loss()

if __name__ == "__main__":
    main()