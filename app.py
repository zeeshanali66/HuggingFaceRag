from flask import Flask, request, jsonify
from gradio_client import Client

# Initialize the Flask app
app = Flask(__name__)

# Gradio API client
client = Client("zeeshanali66/RagPsychologistapi")

# Endpoint to interact with the Gradio model via Flask API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract message from the incoming JSON request
        data = request.get_json()
        message = data.get('message', '')

        # Make prediction using Gradio API
        result = client.predict(message=message, api_name="/predict")

        # Return the result in JSON format
        return jsonify({"response": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
