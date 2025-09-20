# 📘 Study Assistant (Streamlit App)

[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🔗 Project Overview
The **Study Assistant** is a Streamlit-based application that helps students prepare for exams by:  
- Extracting answers directly from PDFs  
- Generating **summaries** for quick revision  
- Auto-creating **MCQs** for practice  

It transforms lengthy notes or textbooks into concise, exam-ready material 🚀

---

## 🚀 Features
- 📂 **Upload PDFs** – class notes, syllabus, or textbooks  
- 🔎 **Question Answering** – ask a question, get top passages + a well-framed answer  
- ✍️ **Summarization** – generate short, clear summaries for revision  
- 📝 **MCQ Generator** – auto-creates multiple choice questions with correct answers & references  
- 🧹 **PDF Cleaning** – removes noisy headers/footers automatically  
- ⚡ **Lightweight & Local** – runs entirely on your system with Streamlit  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Streamlit** – interactive web app framework  
- **NLTK** – natural language processing utilities  
- **FAISS** – vector similarity search  
- **SentenceTransformers** – semantic embeddings  

---

## 📂 Repository Structure
study_assistant/
│── app.py # Main Streamlit app
│── helper.py # PDF processing, embeddings, QA, and MCQ logic
│── requirements.txt # Python dependencies
│── data/ # (optional) sample PDFs
│── vectorstore/ # auto-generated FAISS DB (ignored in git)

yaml
Copy code

---

## ⚡ Installation & Usage

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yashrjgrg8630/study-assistant.git
cd study-assistant
2️⃣ Create and activate a virtual environment (Windows)

bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies

bash
Copy code
pip install -r requirements.txt
4️⃣ Run the application

bash
Copy code
streamlit run app.py
👉 Open http://localhost:8501 in your browser.

📌 Notes
Ensure you have Python 3.10+ installed

The vectorstore/ folder will be auto-generated for embeddings (already ignored in .gitignore)

For large PDFs, processing time may vary depending on hardware

🎯 Future Enhancements
📄 Support for Docx/Markdown in addition to PDFs

📝 Advanced exam practice modes (True/False, Fill in the Blanks)

☁️ Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

🤝 Contributing
Contributions, issues, and feature requests are welcome!

Fork the repository

Create your feature branch → git checkout -b feature/YourFeature

Commit changes → git commit -m "Add feature"

Push to branch → git push origin feature/YourFeature

Open a Pull Request

📜 License
This project is licensed under the MIT License – feel free to use and modify.
