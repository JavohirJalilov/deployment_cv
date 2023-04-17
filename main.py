# prediction model with alexnet
from PIL import Image
from model import (
    model_load,
    preprocess_input,
    to_cuda,
    predict,
    get_class
)

image = Image.open('chair.jpg')
model = model_load()
input_batch = preprocess_input(image)
model = to_cuda(model)
probabilities = predict(model, input_batch)
class_index = get_class(probabilities)

print(class_index)

