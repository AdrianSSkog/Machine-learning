from torchvision.io import decode_image, read_file
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

imgNetPath = r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\imagenet_class_index.json"
image_paths = [
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\Yorkshire-Terrier.jpg",
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\great+white+shark2.jpg",
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\lynx.jpg",
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\Chewbacca.jpg",
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\dogfish.jpg",
    r"C:\Users\adria\Desktop\Python\Machine-learning\labb2\data\lynxFurrJacket.jpg"   
]
test_images = [
    {"path" : image_paths[0], "label" : "Yorkshire terrier", "type" : "positive"},
    {"path" : image_paths[1], "label" : "great white shark", "type" : "positive"},
    {"path" : image_paths[2], "label" : "lynx", "type" : "positive"},
    {"path" : image_paths[3], "label" : "chewbacca", "type" : "negative"},
    {"path" : image_paths[4], "label" : "dogfish", "type" : "negative"},
    {"path" : image_paths[5], "label" : "furr jacket", "type" : "negative"}
]

def get_model_and_weights(mod="resnet18"):
    weights = get_model_weights(mod).DEFAULT
    model = get_model(mod, weights=weights).eval()
    return model, weights

def image_preprocess(img_path, weights):
    img_bytes = read_file(img_path)
    img = decode_image(img_bytes)
    
    preprocess = weights.transforms()

    return img, preprocess(img)

def get_attribution_map(input_tensor, model, layer="layer4"):
    cam_extractor = LayerCAM(model, target_layer=layer)
    out = model(input_tensor.unsqueeze(0))
    return cam_extractor(out.squeeze(0).argmax().item(), out)

def normalize_and_resize_cam(activation_map, pil_image):
    activation_map = activation_map[0]
    cam = activation_map.squeeze(0)

    if cam.ndim == 3:
        cam = cam.squeeze(0)

    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    cam_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1],
        mode="bilinear",
        align_corners=False
    ).squeeze()

    cam_np = cam_resized.cpu().numpy()

    pil_activation_map = to_pil_image(cam_resized, mode="F")

    return cam_np, pil_activation_map


def get_pred(input_tensor, model):
    logits = model(input_tensor.unsqueeze(0))
    preds = logits.softmax(dim=1)
    return preds.squeeze(0).detach()

def load_class_index(path):
    with open(path, "r") as file:
        class_index = json.load(file)
    return class_index

def predict_class(output_tensor, class_index):   
    probs = output_tensor.squeeze()

    top_index = int(probs.argmax())

    synset_id, class_name = class_index[str(top_index)]

    return {
        "class index" : top_index,
        "class id" : synset_id,
        "class name" : class_name,
        "confidence" : float(probs[top_index])
    }

def top_k_predictions(output_tensor, class_index, top_k=5):
    probs = output_tensor.squeeze()
    top_k_probs = probs.topk(top_k)

    result = []

    for i in range(top_k):
        idx = int(top_k_probs.indices[i])
        class_name = class_index[str(idx)][1]
        confidence = float(top_k_probs.values[i])
        result.append((class_name, confidence))
    return result 

def plot_cam(image_paths, model, weights):
    n = len(image_paths)
    fig, ax = plt.subplots(n, 3, figsize=(9, 4*n))

    for j in range(n):
        image, input_tensor = image_preprocess(image_paths[j], weights)

        attribution_map = get_attribution_map(input_tensor, model)

        pil_image = to_pil_image(image)

        cam_np, pil_attribution_map = normalize_and_resize_cam(attribution_map, pil_image)
        
        overlay = overlay_mask(pil_image, pil_attribution_map, alpha=0.5)
        
        ax[j][0].imshow(pil_image)
        ax[j][1].imshow(cam_np, cmap="jet")
        ax[j][2].imshow(overlay)

        titles = ["Original", "attribution map", "overlay"]

        for i in range(3):
            ax[j][i].axis("off")
            ax[j][i].set_title(titles[i])
    plt.tight_layout()
    plt.show()

def plot_layers(image_path, model, weights, layers=["layer1", "layer2", "layer3", "layer4"]):
    image, input_tensor = image_preprocess(image_path, weights)
    pil_image = to_pil_image(image)

    n = len(layers)
    fig, ax = plt.subplots(1, n+1, figsize=(15, 5))

    ax[0].imshow(pil_image)
    ax[0].set_title("Original")
    ax[0].axis("off")

    for i, layer in enumerate(layers):
        attribution_map = get_attribution_map(input_tensor, model, layer=layer)
        pil_attribution_map = normalize_and_resize_cam(attribution_map, pil_image)[1]
        overlay = overlay_mask(pil_image, pil_attribution_map, alpha=0.5)
        ax[i+1].imshow(overlay, cmap="jet")
        ax[i+1].set_title(layer)
        ax[i+1].axis("off")
    plt.tight_layout()
    plt.show()
        

def print_prediction(input_tensor, model, class_index):
    pred = get_pred(input_tensor, model)
    prediction = predict_class(pred, class_index)

    for key, value in prediction.items():
        print(f"{key} : {value}")



def prediction_summary(class_index, model, weights, test_images):
    for img in test_images:
        print("\n")
        print("="*20)
        print(f"Class label: {img['label']}")
        print(f"type: {img['type']}")

        input_tensor = image_preprocess(img["path"], weights)[1]
        output_tensor = get_pred(input_tensor, model)
        top5_preds = top_k_predictions(output_tensor, class_index)

        print(f"Predicted class: {top5_preds[0][0]}, Confidence: {top5_preds[0][1]} ")
        print("")
        print("top 5:")
        
        for pred in top5_preds:
            print(f"{pred[0]} : {pred[1]}")

def plot_test_images(test_images):
    n = len(test_images)
    fig, ax = plt.subplots(2, int(n/2), figsize=(8, 4))
    for i in range(n):
        img_info = test_images[i]
        img_bytes = read_file(img_info["path"])
        img = decode_image(img_bytes)
        pil_img = to_pil_image(img)
        if img_info["type"] == "positive":
            r = 0
            c = i
        else:
            r = 1
            c = i - 3
        ax[r][c].imshow(pil_img)
        ax[r][c].set_xticks([])
        ax[r][c].set_yticks([])
        ax[r][c].set_title(img_info["label"])
        ax[r][c].set_xlabel(img_info["type"])
    plt.tight_layout()
    plt.show()
