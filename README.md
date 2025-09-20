📘 Study Assistant (Streamlit App)

A Streamlit-based Study Assistant that helps students extract answers, generate summaries, and create practice MCQs directly from their syllabus or notes PDFs.
The app is designed to make exam preparation faster by transforming lengthy notes or textbooks into concise, exam-ready material.

🚀 Features

📂 Upload PDFs – class notes, syllabus, or textbooks.

🔎 Question Answering – ask a question, get top passages + a well-framed answer.

✍️ Summarization – generate short, clear summaries for revision.

📝 MCQ Generator – auto-creates multiple choice questions with correct answers & references.

🧹 PDF Cleaning – removes noisy headers/footers automatically.

⚡ Lightweight & Local – runs entirely on your system with Streamlit.

🛠️ Tech Stack

Python 3.10+

Streamlit – interactive web app framework

NLTK – natural language processing utilities

FAISS – vector similarity search

SentenceTransformers – semantic embeddings

📂 Project Structure
study_assistant/
│── app.py             # Main Streamlit app
│── helper.py          # PDF processing, embeddings, QA, and MCQ logic
│── requirements.txt   # Python dependencies
│── data/              # (optional) sample PDFs
│── vectorstore/       # auto-generated FAISS DB (ignored in git)

⚡ Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/yashrjgrg8630/study-assistant.git
cd study-assistant

2️⃣ Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run app.py


👉 Then, open http://localhost:8501
 in your browser.

📌 Notes

Ensure you have Python 3.10+ installed.

The vectorstore/ folder will be auto-generated for embeddings (already ignored in .gitignore).

For large PDFs, processing might take additional time depending on your hardware.

🎯 Future Enhancements

Support for Docx/Markdown in addition to PDFs

Advanced exam practice modes (True/False, Fill in the Blanks)

Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

🤝 Contributing

Contributions, issues, and feature requests are welcome!

Fork the repository

Create your feature branch →

git checkout -b feature/YourFeature


Commit changes →

git commit -m "Add feature"


Push to branch →

git push origin feature/YourFeature


Open a Pull Request

📜 License

This project is licensed under the MIT License – feel free to use and modify.
