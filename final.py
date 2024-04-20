from flask import Flask, request, send_file, render_template
import io
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, detection
import json
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

app = Flask(__name__)

# Load a pre-trained ResNet50 model for image classification
model_classification = resnet50(pretrained=True)
model_classification.eval()

# Load a pre-trained Faster R-CNN model for object detection
model_detection = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_detection.eval()

# Define the transformation for the classification model
transform_classification = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the transformation for the detection model
transform_detection = transforms.Compose([
    transforms.ToTensor(),  # Ensure the pixel range is [0, 1]
])

# Load ImageNet labels
imagenet_labels = json.load(open('imagenet-simple-labels.json'))

# Load the model for image captioning
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Prepare the image for classification
    input_tensor_classification = transform_classification(image)
    input_batch_classification = input_tensor_classification.unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        classification_result = model_classification(input_batch_classification)
        predicted_class = classification_result.argmax(dim=1).item()
        class_label = imagenet_labels[predicted_class]

    # Prepare the image for detection
    input_tensor_detection = transform_detection(image)
    input_batch_detection = input_tensor_detection.unsqueeze(0)

    # Detect objects
    with torch.no_grad():
        detection_result = model_detection(input_batch_detection)

    # Check if detections are found
    if detection_result[0]['boxes'].nelement() == 0:
        return "No objects detected", 400

    # Generate caption for the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = captioning_model.generate(**inputs)
    caption = captioning_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Draw results
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 15)

    # Draw bounding boxes and labels for detected objects
    for box, label, score in zip(detection_result[0]['boxes'], detection_result[0]['labels'], detection_result[0]['scores']):
        if score > 0.5:
            box = box.int().tolist()
            label_name = imagenet_labels[label.item()]
            draw.rectangle(box, outline="red", width=8)
            draw.text((box[0], box[1] - 10), f"{label_name}: {score:.2f}", fill="red", font=font)

    # Add classification and caption
    draw.text((10, 10), f"Class: {class_label}", fill="blue", font=font)
    draw.text((10, 30), f"Caption: {caption}", fill="white", font=font)

    # Save and send the modified image
    modified_image_bytes = io.BytesIO()
    image.save(modified_image_bytes, format="JPEG")
    modified_image_bytes.seek(0)

    return send_file(modified_image_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
