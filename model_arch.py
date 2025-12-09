import torch.nn as nn

class AlzheimerDetector(nn.Module):
    """
    Model architecture replicates TinyVGG model
    from CNN explainer website
    https://poloclub.github.io/cnn-explainer/

    This CNN-based architecture is designed for Alzheimer's disease stage
    classification from MRI images.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, image_dimension: int):
        """
        Initialize the AlzheimerDetector model.

        Args:
            input_shape (int): Number of input channels (3 for RGB images)
            hidden_units (int): Number of hidden units in convolutional layers
            output_shape (int): Number of output classes (4 for Alzheimer's stages)
            image_dimension (int): Input image dimension (128x128)
        """
        super().__init__()

        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * image_dimension // 2 // 2 * image_dimension // 2 // 2,
                      out_features=output_shape)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of MRI images

        Returns:
            Output tensor with class predictions
        """
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

