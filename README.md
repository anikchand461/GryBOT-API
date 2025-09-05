# GryBOT-API

GryBOT-API is a FastAPI-based chatbot backend powered by **Google Gemini API**.  
It provides an easy-to-use REST endpoint where clients (web apps, mobile apps, or other backends) can send queries and receive AI-generated responses.

---

## 🔑 Features
- Simple **REST API** endpoint for chatbot conversations  
- Integrates with **Google Gemini API**  
- Built with **FastAPI** (async, high-performance)  
- Supports JSON request/response format  
- Easy deployment on **Render** or any cloud platform  

---

## ⚙️ Tech Stack
- **FastAPI** (Python 3.11+)  
- **httpx** (async HTTP client)  
- **dotenv** (for environment variable management)  
- **Google Gemini API**  

---

## 📡 How to Use GryBOT API

The API exposes a single endpoint that accepts a user query and responds with an AI-generated answer.

---

### 🔹 Endpoint

POST https://grybot-api.onrender.com/chat/


---

### 🔹 Headers
| Key              | Value                        | Required |
|------------------|------------------------------|----------|
| `Content-Type`   | `application/json`           | ✅        |
| `gemini-api-key` | Your Google Gemini API key   | ✅        |

---

### 🔹 Request Body
Send a JSON object with the query:

```json
{
  "query": "Hello GryBOT, how are you?"
}
```

### Example cURL request

curl -X POST "https://grybot-api.onrender.com/chat/" \
     -H "Content-Type: application/json" \
     -H "gemini-api-key: YOUR_GEMINI_API_KEY" \
     -d '{"query": "Tell me a joke"}'


### Example Python usage

```python
import requests

url = "https://grybot-api.onrender.com/chat/"
headers = {
    "Content-Type": "application/json",
    "gemini-api-key": "YOUR_GEMINI_API_KEY"
}
data = {"query": "What is AI?"}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

