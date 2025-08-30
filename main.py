from flask import request,jsonify,Flask
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("lakshayjain2233/toxicity-model")
model = AutoModelForSequenceClassification.from_pretrained("lakshayjain2233/toxicity-model")
# Set model to evaluation mode
model.eval()

app = Flask(__name__)
CORS(app)


def batch_predict(batch_texts : list):
    outputs = []

    print("Batch Prediction Results:")
    print("=" * 60)

    # Tokenize all texts at once for batch processing
    batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Run batch inference
    with torch.no_grad():
        batch_outputs = model(**batch_inputs)
        batch_predictions = torch.nn.functional.softmax(batch_outputs.logits, dim=-1)
        predicted_classes = torch.argmax(batch_predictions, dim=-1)
        confidences = torch.max(batch_predictions, dim=-1).values

    for i, (text, pred_class, confidence, probs) in enumerate(zip(batch_texts, predicted_classes, confidences, batch_predictions)):
        print(f"Text {i+1}: {text}")
        print(f"Predicted class: {pred_class.item()}")
        print(f"Confidence: {confidence.item():.4f}")
        print(f"Class probabilities: [Class 0: {probs[0].item():.4f}, Class 1: {probs[1].item():.4f}]")
        print("-" * 50)
        outputs.append(
            {
                "text": text,
                "predClass":pred_class.item(),
                "confidence": confidence.item()
            }
        )
    return outputs


@app.route("/api/v1/toxicity",methods = ["POST"])
def predict():
    data = request.json
    msg = data.get('msg', '')
    outputs = batch_predict(msg)
    print(outputs)
    return jsonify(outputs)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

