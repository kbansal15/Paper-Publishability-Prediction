# KHARAGPUR DATA SCIENCE HACKATHON
# TEAM: *NEURAL NAVIGATORS*
### Members:
1. Mauryavardhan Singh(Leader)
2. Shubham Kumar Mandal
3. Hardik Mahawar
4. Ayushman Paul

## Task 1: Paper Publishability Prediction 

### Introduction
This project implements a machine learning-based system to predict the publishability of academic or research papers. It uses natural language processing and machine learning techniques to analyze various aspects of research papers and determine their likelihood of being publishable.

### Features
- PDF text extraction and processing
- Comprehensive feature extraction including:
  - Structural analysis (presence of key sections)
  - Content quality metrics (citations, equations, figures, tables)
  - Readability scores
  - Technical content density
- Document embedding using SPECTER model
- Binary classification using Random Forest
- Detailed performance metrics and evaluation

### Prerequisites
Ensure you have the following installed: 
- spacy
- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- PyPDF2
- textstat
- transformers
- langchain
- faiss-cpu
- streamlit
- torch
- networkx
- plotly
- ollama

### Required Libraries

To install all the dependencies for this project, run the following command:

```bash
pip install spacy sentence-transformers scikit-learn pandas numpy matplotlib seaborn PyPDF2 textstat transformers langchain faiss-cpu streamlit torch networkx plotly ollama

python -m spacy download en_core_web_sm
```

### Project Structure For Task 1
```
project/
├── KDSH_Task_1.ipynb     # Main Jupyter notebook containing all code
├── *.pdf                # 15 labeled PDF papers directly in the project folder
├── Papers/               # Directory containing 135 unlabeled PDF papers for analysis
│   └── *.pdf            # Unlabeled PDF files to be analyzed
└── results.csv          # Output file containing classification results

```

### Installation and Setup
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [project-directory]
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Prepare your data:
   - Place PDF papers to be analyzed in the `Papers/` directory
   - Ensure training data (labeled papers) are properly organized as per the format in `main()`

### Usage
1. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook KDSH_Task_1.ipynb
   ```

2. The notebook contains several key functions:
   - `read_pdf()`: Extracts text from PDF files
   - `preprocess_text()`: Cleans and prepares text for analysis
   - `extract_features()`: Generates numerical features from paper text
   - `generate_embedding()`: Creates document embeddings
   - `train_classifier()`: Trains the model on labeled data
   - `predict_paper()`: Makes predictions on new papers

3. The main execution will:
   - Train the model using labeled examples
   - Process all papers in the Papers/ directory
   - Generate predictions
   - Save results to results.csv

### Output
The system generates a `results.csv` file containing:
- Paper ID (filename)
- Publishability prediction (0 or 1)

### Model Performance
The system evaluates performance using:
- F1 Score
- Precision
- Recall
- Detailed classification report

## Task 2: Research Paper Classification

## Introduction
This project aims to classify research papers into specific categories using machine learning models. The system supports both static data classification and real-time dynamic classification with interactive data streaming using Streamlit.

## Features
- **Static Data Classification**: Analyze pre-existing datasets to classify research papers into categories.
- **Real-time Classification**: Stream and classify research papers dynamically in real time.
- **Interactive User Interface**: Streamlit-powered interface for ease of use.

## Project Structure For Task 2
```
project/
├── KDSH_Task2_Stream.py  # Script for real-time data classification.
├── data/                 # Folder containing example datasets (if applicable)
│                         
└── KDSH_Task2_Static.py  # Script for static data classification.
``` 


## How to Run
### File 1: Static Data Classification
1. Navigate to the project folder:
   ```bash
   cd <path_to_project>
   ```
2. Run the .ipynb file 

### File 2: Real-Time Data Classification
1. Navigate to the real-time data folder:
   ```bash
   cd <path_to_project>
   ```
2. Launch the Streamlit app:
   ```bash
   streamlit run KDSH_Task2_Stream.py
   ```
3. Upload a research paper PDF. The system will:
   - Stream data dynamically.
   - Provide classification results in real time.
   - Offer detailed justifications and analysis.

## Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- LangChain
- Scikit-Learn
- PyTorch
- Llama2
- Sentence Transformers
- PyPDF2
- Plotly
- SPECTER

## Future Scope
- Extend support for additional research domains.
- Enhance model accuracy and speed.
- Integrate more advanced NLP techniques.


