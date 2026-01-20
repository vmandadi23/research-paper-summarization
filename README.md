# 📄 Briefly – Full Research Paper Summarization with LED

Briefly is an end-to-end NLP system that summarizes **entire research papers (PDFs)** using **Longformer Encoder–Decoder (LED)** models.  
The project covers **PDF preprocessing**, **dataset construction**, **fine-tuning LED using LoRA**, **quantitative evaluation with ROUGE**, and a **demo UI for real-time summarization**.

---

## 🚀 Project Overview

**What this project does:**

1. User uploads a research paper PDF
2. Full text is automatically extracted (not just abstract)
3. Text is preprocessed and tokenized
4. A fine-tuned **LED model** generates a summary
5. Summary is displayed via a **Gradio UI**
6. Model quality is measured using **ROUGE scores**

---

## 🗂️ Project Structure

Briefly_FullPaper/
│
├── src/
│ ├── extract_fulltext.py # Extract full paper text from PDFs
│ ├── save_arxiv_metadata.py # Fetch abstracts from arXiv
│ ├── build_json_from_extracted.py # Build JSONL dataset
│ ├── train_led.py # Supervised fine-tuning with LoRA
│ ├── train_led_scst.py # (Optional) Reward-based training
│ └── split_json.py # split json into 2 sets
│
├── data/
│ ├── data_raw_pdfs/ # Original PDFs
│ ├── processed/
│ │ ├── extracted/ # Extracted full text (.txt)
│ │ └── dataset/ #contains dataset in json
| |── metadata
│   ├── arxiv_metadata.csv
│ 
│
├── models/
│ └── led_large_lora_final_adapter/ #adapter+tokenizer
│
│
├── notebooks/
│ ├── LED_DEMO.ipynb # Inference & demo notebook
│ └── Led_Large_training.ipynb # LED-Large training notebook
│
├── requirements.txt
└── README.md

---

## 📊 Dataset Details

- **Source**: Local research paper PDFs
- **Naming**: PDFs follow arXiv ID format (e.g., `2512.00580v2.pdf`)
- **Abstracts**: Retrieved from arXiv metadata
- **Used Papers**: around 3000 full papers in phases

### Dataset Split - final phase
| Split | Count |
|-----|------|
| Train | ~1342 |
| Validation | ~150 |



## 📓 Jupyter Notebooks

### 🧪 `Led_Large_training.ipynb`
- End-to-end **GPU training** on Google Colab  
- Dataset loading and preprocessing  
- **LoRA fine-tuning** of LED-Large  
- Model checkpoint and adapter saving  

### 🔍 `LED_DEMO.ipynb`
- Load the **trained LED + LoRA adapter**  
- Run **inference on full research papers**  
- Compute **ROUGE-1 / ROUGE-2 / ROUGE-L** scores  
- Visualize generated summaries vs references  

## ⚙️ Installation & Setup

### 1️⃣ Prerequisites
Make sure you have the following installed:

- Python **3.9 – 3.11**
- NVIDIA GPU (optional, required for fast inference/training)
- Google Colab account (for LED-Large training)

---

### 2️⃣ Create Virtual Environment (Local)
python -m venv .venv

Activate it:
- Windows- .venv\Scripts\activate
- Linux / macOS- source .venv/bin/activate

### 3#️⃣Install Dependencies
pip install --upgrade pip

pip install -r requirements.txt


### Step 1: Put pdfs into one folder
- Copy all research paper PDFs into: data/raw_pdfs/
- PDF names must look like: 2512.00580v2.pdf

### Step 2: Extract Full Text from PDFs
- python src/extract_fulltext.py


### Get Abstracts from arXiv
python src/save_arxiv_metadata.py
- ✔ Fetches abstracts using arXiv IDs
- ✔ Saves metadata in data/processed/metadata/

### Build Training Dataset
 python src/build_json_from_extracted.py
- ✔ Matches full paper text + abstract

### Train the Model (GPU / Colab Recommended)
python src/train_led.py
- ✔ Fine-tunes LED using LoRA
- ✔ Saves model to:
- led_ckpt/final_adapter/

### Validate Model Performance
- validate model from jupyter notebook file - Led_Large_training.ipynb
- ✔ Compares baseline vs fine-tuned model
- ✔ Prints ROUGE-1 / ROUGE-2 / ROUGE-L scores

### Run the Demo (Upload PDF → Get Summary)
- run the demo from LED_DEMO.ipynb










