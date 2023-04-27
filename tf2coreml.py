import coremltools as ct
import tensorflow as tf
import argparse
import os


ap = argparse.ArgumentParser()

ap.add_argument("--input_model", help="Tensorflow model path", required=True)
ap.add_argument("--output_model", help="File of output model", required=True)
ap.add_argument("--is_dynamic", type=int, help="Ask if model has dynamic input. Pass 1 if dynamic inout else pass 0", default=0)
ap.add_argument("--output_dir", help="Directory to save the model. By default saves in current directory", default=".")



def coreml_model_converter(path_to_tf: str, save_dir: str, coreml_file_name: str, dynamic: bool=False):
    """Function to convert tensorflow model to coreml format

    Args:
        path_to_tf (str): Path of the tensorflow model
        save_dir (str): The directory to save coreml model
        coreml_file_name (str): The output model file name
        dynamic (bool, optional): Specifies if model has dynamic inputs (Allowed values are True and False).If True then model is converted using dynamic input shapes. Defaults to False.
    """
    model = tf.keras.models.load_model(path_to_tf)
    input_name = model.inputs[0].name

    if dynamic:
        height = int(input("Please enter lower bound for height :- "))
        width = int(input("Please enter lower bound for width :- "))
        print()

        input_shape = ct.Shape(shape=(ct.RangeDim(lower_bound=1, upper_bound=-1),
                            ct.RangeDim(lower_bound=height, upper_bound=-1),
                            ct.RangeDim(lower_bound=width, upper_bound=-1),
                            3))

    else:
        print("Model is not dynamic")
        height = int(input('Please specify the height :- '))
        width = int(input('Please specify the width  :- '))
        print()

        input_shape = ct.Shape(shape=(ct.RangeDim(lower_bound=1, upper_bound=-1),
                                height,
                                width,
                                3))

    c_model = ct.convert(model, inputs=[ct.TensorType(shape=input_shape, name=input_name)])
    save_pth = os.path.join(save_dir, coreml_file_name) 
    c_model.save(save_pth)




if __name__ == "__main__":
    arg = vars(ap.parse_args())
    print()
    path_to_tf = arg.get('input_model')
    save_file = arg.get('output_model')
    save_dir = arg.get('output_dir', "./")

    dynamic = arg.get('is_dynamic')
    if dynamic == 1:
        coreml_model_converter(path_to_tf, save_dir, save_file, dynamic=True)
    elif dynamic == 0:
        coreml_model_converter(path_to_tf, save_dir, save_file)
    else:
        raise ValueError("Keys 0 and 1 are allowed values for is_dynamic")
    
