Clinic FAQ RAG Agent — Quick Start Guide
This project uses FastAPI + LangChain + DeepSeek + ChromaDB to answer clinic FAQs with RAG.

Follow these steps to run the project.
1. Create Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

3. Add DeepSeek API Key
Create a file named .env in the project root:
DEEPSEEK_API_KEY=your_real_key_here

4. Run the Application
uvicorn app:app --reload

5. Open API Documentation
Swagger UI:
http://127.0.0.1:8000/docs