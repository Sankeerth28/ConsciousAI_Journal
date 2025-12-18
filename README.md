## 🧠 ConsciousAI Journal  
Live Demo: https://huggingface.co/spaces/Sankeerth004/conscious-ai-journal

ConsciousAI is a smart journaling application designed to help you explore your thoughts and feelings. It uses AI to provide empathetic and insightful responses, analyze emotional trends, and help you gain a deeper understanding of your inner world. It's a private, non-judgmental space for self-reflection and growth.

<img width="1902" height="933" alt="Screenshot 2025-07-21 102224" src="https://github.com/user-attachments/assets/2a2b0425-d1ed-447b-94cc-3c73a0a5f806" />

---

## ✨ Features

* **AI-Powered Responses:** Get dynamic, context-aware feedback on your journal entries.
* **Emotion & Value Analysis:** Automatically detects emotions (e.g., happy, sad, anxious) and core values (e.g., growth, honesty, courage) in your writing.
* **Personalized Personas:** Choose from different AI personalities (Supportive, Therapist-like, Coach, Neutral) to tailor the conversation.
* **Long-Term Memory:** Ask questions about your past entries (e.g., "When was the last time I felt hopeful?").
* **Data Analytics:** Visualize your emotional trends and core value themes over time with interactive charts.
* **Journal Export:** Download your entire journal as a CSV file at any time.
* **Streak Tracker:** Stay motivated with a daily journaling streak counter.

---

## 💻 Hardware Requirements & Environment

This project was developed and tested in **Google Colab** using a T4 GPU. The language models used are large and computationally intensive.

* **GPU Required:** To run this project on your own computer, a dedicated NVIDIA GPU with sufficient VRAM is necessary. The `bitsandbytes` library is used for 4-bit model quantization, which requires a compatible CUDA-enabled GPU.
* **Google Colab Recommended:** If you do not have a powerful local GPU, it is highly recommended to run this notebook in Google Colab. They provide free access to GPUs, which is sufficient to run this application smoothly.

---

## 🛠️ Tech Stack

This project leverages a modern stack of AI and data science technologies to deliver its features.

* **Core Language:**
    * **Python:** The primary language for all backend logic and model orchestration.

* **AI & Machine Learning:**
    * **Hugging Face Transformers:** For loading and running state-of-the-art pre-trained models.
    * **LangChain:** Used as a framework to chain together AI components, manage prompts, and integrate the language model with the vector database.
    * **FAISS (Facebook AI Similarity Search):** An efficient library for similarity search, used here as the vector database for the journal's memory.
    * **PyTorch:** The deep learning framework on which the models operate.

* **Models Used:**
    * **LLM (for response generation):** `google/flan-t5-large`
    * **Classifier (for emotion/value detection):** `facebook/bart-large-mnli`
    * **Embedding Model (for text vectorization):** `sentence-transformers/all-MiniLM-L6-v2`

* **User Interface & Data Handling:**
    * **Gradio:** A fast and easy way to build and share the web-based UI for the application.
    * **Pandas:** For all data manipulation tasks, including handling the journal's CSV log.
    * **Plotly:** For creating the interactive charts and visualizations in the Analytics tab.

---

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine or in Google Colab.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/conscious-ai-journal.git](https://github.com/your-username/conscious-ai-journal.git)
cd conscious-ai-journal
```

*(Replace `your-username` with your actual GitHub username)*

### 2. Create a `requirements.txt` File

Create a file named `requirements.txt` in the project's root directory and add the following lines to it:

```
langchain
langchain-community
faiss-cpu
transformers
sentence-transformers
gradio
pandas
torch
plotly
accelerate
bitsandbytes
```

### 3. Install Dependencies

It's recommended to use a virtual environment to avoid conflicts with other projects.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate it
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

Launch Jupyter Lab or Jupyter Notebook and open the `Organized_ConsciousAI_Notebook.ipynb` file.

```bash
jupyter lab
```

### 5. Log in to Hugging Face

When you run the first code cell (`Block 0`), you will be prompted to enter a Hugging Face access token. You can get a token from your [Hugging Face account settings](https://huggingface.co/settings/tokens). This is required to download the language models used in this project.

---

## 📖 How to Use

1.  **Run the Cells:** Execute all the cells in the notebook in sequential order.
2.  **Launch the App:** The final cell will initialize all the models and launch the Gradio web interface. A public link will be generated for you to open the app in your browser.
3.  **Interact with the UI:**
    * **✍️ Journal Tab:** Write your entry, select an AI persona, and click "Submit". You can then provide feedback on the AI's response to help it learn.
    * **📊 Analytics Tab:** Click "Analyze My Journal" to see visualizations of your emotional data and trends over time.
    * **🧭 Ask Your Journal Tab:** Type a question about your past entries to get an AI-generated summary based on your journal's memory.
