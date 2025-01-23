
import torch
# import pickle
import onnxruntime as rt
from model import model_list

# def torch2pickle(model_checkpoint: str):
#     model_type = model_checkpoint.split("_")[0]
#     model,_ = model_list(model_type)
#     model.load_state_dict(torch.load("models/"+model_checkpoint, weights_only=True))
#     with open(f"models/{model_type}_model.pkl", "wb") as file:
#         pickle.dump(model, file)
#     print("Model converted to pickle")

def torch2onnx(model_checkpoint: str, get_optimized: bool = True):
    model_type = model_checkpoint.split("_")[0]
    model,_ = model_list(model_type)
    model.load_state_dict(torch.load("models/"+model_checkpoint, weights_only=True))
    model.eval()
    dummy_input = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model=model,
        args=(dummy_input,),
        f=f"models/{model_type}_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    if get_optimized:
        optimize_onnx(f"models/{model_type}_model.onnx")
    print("Model converted to onnx")
    

def optimize_onnx(model_onnx_path: str):
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = model_onnx_path

    session = rt.InferenceSession(model_onnx_path, sess_options)


if __name__ == "__main__":
    torch2onnx("cnn_2025-01-22_14-39-37.pth")