import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

try:
    client = genai.Client(api_key=api_key)
    print("✅ Connection successful! Here are your models:")
    print("-" * 30)
    
    # Simple loop that just prints the name
    for model in client.models.list():
        print(f"Model: {model.name}")
            
except Exception as e:
    print(f"❌ Error: {e}")

    print("error us hight")