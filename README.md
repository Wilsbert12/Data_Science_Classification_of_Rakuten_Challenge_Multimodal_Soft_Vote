# Rakuten Multimodal Product Classification

**Academic Context**: This project serves as the capstone for the Data Science module, demonstrating end-to-end machine learning pipeline development from exploratory data analysis through multimodal model deployment.

This project tackles large-scale multimodal (text and image) product classification for Rakuten France's e-commerce platform. The goal is to automatically categorize products into the correct product type codes using both product titles/descriptions and product images.

**Challenge Context**: [Rakuten Data Challenge](https://challengedata.ens.fr/challenges/35)

- **Dataset**: ~99K product listings (84,916 training, 13,812 test)
- **Metric**: Weighted F1-Score
- **Modalities**: Text (French/German titles + descriptions) + Product Images
- **Categories**: 27 product type codes in Rakuten France catalog

**Development Evolution**: This project was developed before formal MLOps training, resulting in a distributed implementation across local development, Google Colab (for compute-intensive training), and Google Drive (for model storage). A follow-up MLOps capstone project addresses the systematic experiment tracking, model versioning, and deployment automation challenges identified during this data science phase.

## 🎯 Business Impact

Product categorization is fundamental for e-commerce marketplaces, enabling:
- Personalized search and recommendations
- Improved query understanding
- Reduced manual cataloging effort
- Better handling of new and used products from various merchants

## 🚀 Technical Approach

### Ensemble Method: Soft Voting Classifier
We implemented a multimodal ensemble combining three specialized models:

| Model | Modality | F1-Score | Description |
|-------|----------|----------|-------------|
| **SVM** | Text | **0.76** | TF-IDF vectorized features with French preprocessing |
| **CamemBERT** | Text | **0.75** | French transformer with minimal preprocessing |
| **VGG16** | Image | **0.85** | CNN transfer learning (best individual model) |
| **Ensemble** | Multimodal | **TBD** | Soft voting across all modalities |

### Data Processing Pipeline

**Language Detection & Translation**: ✅ **Pre-completed**
- **Status**: Translation pipeline complete - no need to re-run
- **Coverage**: 23.4% non-French content translated using DeepL API
- **Implementation**: Available in `notebooks/reference/DeepL.ipynb` and `utils/loc_utils.py`
- **Note**: Requires personal DeepL API key. Results pre-computed and cached.

**Text Preprocessing**:
- **Base processing**: Combines designation + description gracefully handling 35% missing descriptions
- **Model-specific optimization**:
  - **Classical ML**: Accent normalization + French stop words + TF-IDF vectorization
  - **BERT**: Minimal preprocessing using CamemBERT native tokenizer
- **Translation integration**: Uses pre-computed French translations for consistent processing

**Key Optimizations**:
- ✅ Preprocessing efficiency with shared base processing
- ✅ Translation ready with cached results for immediate use
- ✅ Notebook independence for reproducible execution

## 📁 Repository Structure

```
├── 📊 data/                          # All datasets and preprocessing artifacts
│   ├── raw/                          # Original immutable datasets
│   │   ├── X_train.csv              # Training features (text + image IDs)
│   │   ├── X_test.csv               # Test features
│   │   └── y_train.csv              # Training labels (product type codes)
│   ├── language_analysis/           # Language detection and translation pipeline
│   │   ├── df_langdetect.csv        # Language detection results
│   │   ├── df_localization.csv      # Final dataset with DeepL translations
│   │   └── ...                      # Translation processing artifacts
│   └── prdtypecode_to_category_name.json  # Category mapping reference
├── 🧠 models/                        # Trained model artifacts
│   ├── bert/                         # CamemBERT model files
│   ├── svm_classifier.pkl            # Trained SVM model
│   └── vgg16_transfer_model.pth      # VGG16 with custom classification head
├── 📓 notebooks/                     # All Jupyter notebooks (consolidated)
│   ├── 01_exploratory_data_analysis.ipynb  # Main EDA with hypothesis framework
│   ├── 02_language_analysis.ipynb          # Language detection and translation
│   ├── 03_classical_ml_text.ipynb          # SVM training pipeline
│   ├── 05_Image_classification.ipynb       # VGG16 training
│   └── reference/                           # Translation utilities
│       ├── DeepL.ipynb                     # Complete translation workflow
│       └── ...
├── 🛠️ utils/                         # Utility functions
│   ├── text_utils.py                 # Text preprocessing and cleaning
│   ├── image_utils.py                # Image processing utilities
│   └── loc_utils.py                  # Localization and translation utilities
├── 🌐 pages/                         # Streamlit application pages
├── 🎨 assets/                        # Logos and visualizations
├── votingClassifier.py               # Ensemble implementation and inference
└── *.py                              # Streamlit app components
```

## 🔧 Installation & Usage

### Prerequisites
- Python 3.11 (recommended for ML package compatibility)
- Git
- Homebrew (for Mac users)

### Setup

**Install Python 3.11 (Mac)**
```bash
# Install Python 3.11 for better ML package compatibility
brew install python@3.11
```

**Create and Setup Environment**
```bash
# Clone repository
git clone [repository-url]
cd rakuten-multimodal-classification

# Create virtual environment with Python 3.11
python3.11 -m venv rakuten-env

# Activate virtual environment
source rakuten-env/bin/activate  # Mac/Linux
# or
rakuten-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=rakuten-env --display-name="Python (rakuten-env)"
```

**For Jupyter Notebooks**
- **VS Code**: Select the Python 3.11 interpreter from your rakuten-env folder
- **Jupyter Lab**: `jupyter lab` (after activating environment)
- **Any IDE**: Configure to use `rakuten-env/bin/python3.11`

### Running the Application

```bash
# Ensure virtual environment is activated
source rakuten-env/bin/activate  # Mac/Linux

# Run Streamlit app
streamlit run Home.py

# Run API service (if available)
uvicorn api.main:app --reload
```

**Deactivating Environment**
```bash
deactivate
```

## 📊 Key Features

### Data Science Pipeline
- ✅ **Data Exploration**: Product distribution, missing data analysis, multimodal insights
- ✅ **Text Processing**: Dual preprocessing pipelines for Classical ML vs BERT approaches
- ✅ **Language Detection**: Automated French/non-French classification using langdetect
- ✅ **Translation Integration**: Pre-processed translations ready for model training
- ✅ **Image Processing**: VGG16 transfer learning with custom classification head
- ✅ **Feature Engineering**: Text combination, cleaning, and vectorization strategies
- ✅ **Model Development**: Individual model training with hyperparameter optimization
- ✅ **Ensemble Method**: Soft voting classifier combining all three modalities
- ✅ **Evaluation**: Cross-validation and performance comparison against benchmarks

### Production Features
- ✅ **Streamlit Demo**: Interactive multimodal prediction interface
- ✅ **FastAPI Service**: RESTful API for real-time classification
- ✅ **Model Persistence**: Trained models and preprocessing pipelines saved as artifacts
- ✅ **Translation Optimization**: Pre-processed translations ready for immediate use
- ✅ **Ensemble Inference**: Soft voting predictions from all three models

### EDA Optimization Achievements
- ✅ **80%+ Code Reduction**: Eliminated analysis bloat while maintaining insights
- ✅ **Hypothesis-Driven Framework**: 6 testable hypotheses (H1-H6) linking data patterns to model performance
- ✅ **Visual Validation**: Wordcloud comparison showing progressive text cleaning stages
- ✅ **Professional Structure**: Self-contained execution with clean, reproducible workflow

## 🔬 Hypotheses Framework

Based on comprehensive data analysis, we established 6 testable hypotheses:

- **H1**: Inter-parent classification is easier than intra-parent classification
- **H2**: Intra-parent classification is more challenging due to shared vocabulary
- **H3**: Image features help with fine-grained distinctions (VGG16 > Text models)
- **H4**: Rare single-subcategory parents achieve high precision despite low samples
- **H5**: Subcategory complexity negatively correlates with classification performance
- **H6**: Large subcategories achieve better F1 scores due to more training examples

## 📈 Performance Results

**Individual Model Performance:**
- **VGG16 (Image)**: 0.85 F1-Score - Best performing individual model
- **SVM (Text)**: 0.76 F1-Score - Classical ML with comprehensive preprocessing
- **CamemBERT (Text)**: 0.75 F1-Score - Transformer with minimal preprocessing

**Ensemble Performance**: TBD (soft voting across all modalities)

## 💻 Technical Stack

- **ML/DL**: scikit-learn, transformers (CamemBERT), PyTorch, TensorFlow/Keras
- **Data Processing**: pandas, numpy, PIL, BeautifulSoup
- **Text Processing**: TF-IDF, CamemBERT tokenizer, langdetect, French stop words
- **Translation**: Pre-processed French translations for multilingual content
- **Computer Vision**: VGG16 transfer learning, torchvision transforms
- **Web Framework**: Streamlit, FastAPI (planned)
- **Model Persistence**: joblib, PyTorch state dictionaries
- **Deployment**: Streamlit Cloud, Python 3.11 virtual environments

## 🏗️ Architecture & Development Notes

**Distributed Implementation**: Due to computational requirements and pre-MLOps development practices:
- **Local Repository**: Exploratory data analysis, classical ML pipeline (SVM), project documentation
- **Google Colab**: BERT/CamemBERT training and ensemble integration
- **Google Drive**: Large model storage and sharing
- **Local Files**: Smaller models (SVM: 29.9MB) included in repository

**MLOps Evolution**: This project represents pre-MLOps development practices. Future improvements through dedicated MLOps capstone will address:
- ❌ Scattered results tracking → ✅ Systematic experiment tracking (MLflow)
- ❌ Manual model versioning → ✅ Automated model versioning  
- ❌ Distributed storage → ✅ Centralized management with streamlined deployment

## 🚀 Deployment

**Streamlit Application**
- **Demo**: Interactive multimodal prediction interface
- **Features**: Model comparison, data exploration, real-time predictions
- **Note**: App may take 30-60 seconds to wake up from Streamlit Cloud sleep mode

**API Service**
- **FastAPI**: RESTful endpoints for production inference
- **Input**: Product text + image upload
- **Output**: Predicted category + confidence scores

## 🔮 Future Improvements

**Technical Enhancements**:
- [ ] Ensemble weight optimization based on validation performance
- [ ] Model performance analysis and feature importance studies
- [ ] Cross-validation results integration
- [ ] Hypothesis validation with actual model F1 scores

**Code Organization**:
- [ ] Consolidate preprocessing pipelines for better maintainability
- [ ] Repository cleanup and file organization optimization
- [ ] Git LFS integration for large model files

**Presentation**:
- [ ] Professional README completion ✅
- [ ] Streamlit app optimization
- [ ] FastAPI documentation
- [ ] Portfolio presentation materials

## 🤝 Contributing

This project was developed as a capstone for a Data Science program, focusing on real-world multimodal classification challenges in e-commerce. 

For contributions:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Rakuten France** for providing the challenge dataset and business context
- **Data Science Program** for academic guidance and technical framework
- **Open Source Community** for the powerful libraries that made this project possible

---

**Note**: This README documents the current state of the project after major optimization and restructuring. The repository represents a complete evolution from initial development to production-ready structure with comprehensive documentation and reproducible workflows.