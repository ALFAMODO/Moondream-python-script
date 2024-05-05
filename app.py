
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import uvicorn

# Initialize FastAPI application
app = FastAPI()

# Configure CORS to allow requests from specific origins (e.g., frontend applications)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the Next.js app
    allow_methods=["GET", "POST"],  # Allow HTTP GET and POST methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-04-02"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

@app.get("/")
async def root():
    """Endpoint to check if the server is running."""
    return {"message": "Welcome to FastAPI"}

@app.post("/process/")
async def process_data(
    prompt: str = Query(...),  # Get 'prompt' from query parameters
    image_link: Optional[str] = Query(None)  # Get 'image_link' from query parameters
):
    """
    Endpoint to process data sent via query parameters.
    Requires 'prompt', and 'image_link' is optional.
    """
    # Check if the required prompt is present
    if not prompt:
        raise HTTPException(status_code=400, detail="The 'prompt' query parameter is required.")

    # Validate the image link
    if not image_link:
        raise HTTPException(status_code=400, detail="The 'image_link' query parameter is required.")

    # Fetch the image from the provided URL
    response = requests.get(image_link)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to retrieve the image.")

    # Load the image with PIL
    image = Image.open(BytesIO(response.content))

    # Encode the image and ask the model to describe it
    enc_image = model.encode_image(image)
    image_description = model.answer_question(enc_image, prompt, tokenizer)

    # Construct the response data
    response_data = {
        "status": "success",
        "prompt": prompt,
        "image_link": image_link,
        "description": image_description,
    }

    # Return the description of the image
    return response_data  # Return the JSON response

if __name__ == "__main__":
    # Start the FastAPI server with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)  # Enable live reloading for development
