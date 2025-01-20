import torch
from torch import nn
from transformers import ResNetForImageClassification
from project_mlops.model import ModelConvolution, hugging_face_resnet


def test_model_initialization():
    """Test if the models can be initialized without errors."""
    model = ModelConvolution()
    assert isinstance(model, nn.Module), "ModelConvolution should be an instance of nn.Module."

    resnet_model = hugging_face_resnet()
    assert isinstance(resnet_model, nn.Module), "ResNet model should be an instance of nn.Module."


def test_forward_pass_convolution():
    """Test the forward pass of the ModelConvolution."""
    model = ModelConvolution()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels (RGB), 224x224 image
    output = model(dummy_input)
    assert output.shape == (1, 53), f"Expected output shape (1, 53), but got {output.shape}."


def test_forward_pass_resnet():
    """Test the forward pass of the Hugging Face ResNet model."""
    resnet_model = hugging_face_resnet()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels (RGB), 224x224 image
    output = resnet_model(dummy_input).logits  # Access logits for classification
    assert output.shape == (1, 53), f"Expected output shape (1, 53), but got {output.shape}."


def test_parameter_count():
    """Test if ModelConvolution has trainable parameters."""
    model = ModelConvolution()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "ModelConvolution should have trainable parameters."


def test_dropout_layers():
    """Test if ModelConvolution has the correct number of dropout layers."""
    model = ModelConvolution()
    dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    assert len(dropout_layers) == 3, f"Expected 3 Dropout layers, but found {len(dropout_layers)}."


def test_resnet_pretrained_weights():
    """Test if the Hugging Face ResNet model loads pretrained weights correctly."""
    resnet_model = hugging_face_resnet()
    assert hasattr(resnet_model, "state_dict"), "Hugging Face ResNet should have a state_dict method."
    state_dict = resnet_model.state_dict()
    assert len(state_dict) > 0, "State dict should not be empty."


def test_invalid_input_shape():
    """Test if ModelConvolution raises an error for invalid input shapes."""
    model = ModelConvolution()
    invalid_input = torch.randn(1, 1, 224, 224)  # Only 1 channel instead of 3
    try:
        _ = model(invalid_input)
        assert False, "Model should raise an error for invalid input shape."
    except RuntimeError:
        pass  # Expected behavior


if __name__ == "__main__":
    # Run tests
    test_model_initialization()
    test_forward_pass_convolution()
    test_forward_pass_resnet()
    test_parameter_count()
    test_dropout_layers()
    test_resnet_pretrained_weights()
    test_invalid_input_shape()
    print("All tests passed!")
