# ReMind.AI

![Project Cover](https://github.com/napilkington/BrainSight-AI/blob/main/img/portada.png?raw=true)

[![Video Presentation](https://img.shields.io/badge/Watch_Presentation-YouTube-red)](https://youtu.be/EnLfzfw9_-8)
[![App Demo](https://img.shields.io/badge/Try_App-Streamlit-blue)](https://brainsight-ai.streamlit.app/)
[![PDF Presentation](https://img.shields.io/badge/Presentation-PDF-orange)](presentation/Presentation_TFM.pdf)

> **Author:** Natalie Pilkington González

## 📋 Project Description

This project is a **classification** exercise to determine the stage of **Alzheimer's disease** from **MRI** (Magnetic Resonance Imaging) scans of patients with this condition.

It uses a **convolutional neural network (CNN) based on the TinyVGG16 architecture** to classify these images into different stages of Alzheimer's disease:
- Mild Impairment
- Moderate Impairment
- No Impairment
- Very Mild Impairment

The project includes Jupyter notebooks for training and prediction, and a **Streamlit application** for easy predictions.

The model achieves high metrics in predicting Alzheimer's stages. For training, the [Best Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data) was used.

The application also features two **AI Assistants** that allow users to obtain medical recommendations and ask specific questions based on the model's prediction. Information is generated through connection with **Gemini, Google's LLM**.

## 🧠 Alzheimer and Computer Vision

**Alzheimer's** is the most common form of dementia that progressively damages brain cells, resulting in:
- Memory and thinking deficits
- Loss of basic skills
- Eventually, death

**Relevant data:**
- Annual treatment cost: 1 trillion USD
- Projection for 2050: 152 million people affected
- Early diagnosis can significantly improve quality of life

**Computer vision** has emerged as a powerful tool in the early detection of Alzheimer's disease, offering a complementary approach to traditional methods. Deep learning techniques such as **CNNs** enable the analysis of large volumes of MRI images with high precision, identifying subtle patterns of brain atrophy and amyloid protein accumulation even in very early stages.

## 📊 Big Data

### Dataset

The dataset used comprises a total of **11,520 images**:
- 10,240 for training
- 1,282 for testing

These images are high-quality magnetic resonance imaging (MRI) scans labeled according to Alzheimer's Disease progression. You can download it from [Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data).

The dataset contains a mix of real and synthetic axial MRIs with four categories:
- **No Impairment**: 100 patients
- **Very Mild Impairment**: 70 patients
- **Mild Impairment**: 28 patients
- **Moderate Impairment**: 2 patients

Each patient's brain was sliced into 32 horizontal axial MRIs, resulting in an extensive and balanced multiclass dataset, especially the training subset.

![Class Distribution](https://github.com/napilkington/BrainSight-AI/blob/main/outputs/class_distribution.png?raw=true)

### Data Preprocessing

The **transforms** module from the **torchvision** library was used to:
- Resize images to 128x128 pixels
- Convert them to PyTorch tensors

The dataset is stored in an **AWS S3 bucket**, accessible through temporary IAM keys provided by an AWS Academy Learner Lab.

### Creating a GAN to Generate Synthetic MRI Images

While the training dataset was extensive and balanced, the test dataset used to evaluate the trained model was much smaller and imbalanced. Due to this and the difficulty of finding real brain MRIs of Alzheimer's patients different from those used in the dataset, a **GAN** was created to generate synthetic images for testing the trained model. Specifically, a **WGAN-GP** (Wasserstein Generative Adversarial Networks with Gradient Penalty) was used, which represents an improvement over conventional GANs, as it significantly reduces problems such as ***Mode Collapse***, which occurs when the generator "stagnates" and learns to produce only a limited subset of modes present in the training data. They also incorporate the ***Gradient Penalty*** function, which uses the Wasserstein distance to improve training stability.

To train the WGAN, a thousand randomly selected images from the test subset of the dataset were used. Once trained, a selection of 200 high-quality synthetic images was extracted and used to test the model in the Streamlit application. The images are located in the **WGAN_Synthetic_Images** folder of this repository. The WGAN thus proved very useful in solving the problem of limited MRI images of Alzheimer's patients due to their confidential nature. Both the images and the models (generator and discriminator) and the Jupyter notebook used are available in the "WGAN" folder of this repository.

**Images from the original dataset**

![Real images from dataset](https://github.com/napilkington/BrainSight-AI/blob/main/img/imagenes_reales.png?raw=true)

**Synthetic images generated by the WGAN**

![Synthetic images generated by WGAN](https://github.com/napilkington/BrainSight-AI/blob/main/img/imagenes_sinteticas.png?raw=true)

## 🔍 Deep Learning with CNN

### Model Architecture

The Deep Learning model comprises two convolutional blocks followed by a fully connected classifier, inspired by the **TinyVGG16** architecture.

#### TinyVGG16 Characteristics:
- Reduced and optimized version of the well-known VGG16 network
- Preserves the fundamental structure: stacked 3x3 convolutional layers, followed by max pooling layers
- Reduces the number of layers and filters compared to VGG16
- Enables faster processing, ideal for real-time applications or with limited computational resources

### Performance Metrics

#### Training Metrics
| Metric | Value |
|---------|-------|
| Loss | 0.0039 |
| Accuracy | 99.90% |
| Precision | 0.9990 |
| Recall | 0.9990 |
| F1-Score | 0.9990 |

#### Test Metrics
| Metric | Value |
|---------|-------|
| Loss | 0.1574 |
| Accuracy | 95.47% |
| Precision | 0.9563 |
| Recall | 0.9547 |
| F1-Score | 0.9548 |

### Performance Visualization

#### Training-Test Loss During Epochs:

![Train Test Loss observed during 20 epochs](https://github.com/napilkington/BrainSight-AI/blob/main/outputs/train-test-loss-over-epochs.png)

#### Confusion Matrices:

**Training Data:**

![Train Data Confusion Matrix](https://github.com/napilkington/BrainSight-AI/blob/main/outputs/train_data_cnf_mat.png)

**Test Data:**

![Test Data Confusion Matrix](https://github.com/napilkington/BrainSight-AI/blob/main/outputs/test_data_cnf_mat.png)

## 🤖 AI Assistants

As a complement to the prediction model, this application includes two **AI Assistants**:

### 1. AI Assistant for Medical Recommendations

Once the prediction is generated, the user can obtain medical recommendations based on the diagnosis. This assistant:

- **Understands context**: Analyzes the diagnosis and additional information provided
- **Accesses medical knowledge**: Uses updated information, clinical guidelines, and research data
- **Generates personalized responses**: Adapts recommendations to the patient's specific situation
- **Interacts conversationally**: Maintains a dialogue with the user answering questions

### 2. Specialized Medical Chatbot

Acts as a virtual assistant specialized in Alzheimer's, allowing the user to ask questions and receive informative answers. Features:

- Captures and displays user questions in the history
- Sends queries to Gemini and presents the responses
- Handles errors during response generation

Both assistants use the **Gemini 1.5 Flash model**, with the API key stored as an environment variable.

## 💻 Streamlit Application

![Streamlit Interface]

The application has three interfaces accessible from the sidebar menu:

### 1. Patient Data
Interface to register and save patient medical data and generate reports.

### 2. Diagnosis
Interface to upload MRI images. The model predicts the Alzheimer's phase and allows obtaining medical recommendations through the AI agent.

### 3. Virtual Assistant
Chatbot to answer medical questions related to Alzheimer's.

**Available at:**
- Local: http://localhost:8501/

## 🔧 How to Get the Code

```bash
git clone https://github.com/Kutcher1945/remind_analysis.git
cd remind_analysis
```

### Create a Virtual Environment
1. Install Python 3.10 if not available
2. Create and activate the virtual environment:

```bash
python3.10 -m venv my_env
.\my_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Configure the Gemini API Key
Add your API key to the `.env` file:

```
GEMINI_API_KEY=your_api_key
```

### Deploy the App with Streamlit (locally)

```bash
streamlit run app.py
```

## 🧪 Testing the Model

The "WGAN_Synthetic_Images" folder contains MRI images generated with a GAN from the original dataset. Use these images to test the model's performance through the Streamlit application or the provided Jupyter notebooks.

## 📁 Project Structure

```
root
│
├── img                          # Project illustrative images
├── models                       # Trained models
│   └── alz_CNN.pt               # PyTorch model for classification
├── notebooks                    # Jupyter notebooks
│   ├── Alzehmier_CNN.ipynb      # Notebook for training
│   ├── data_explore.ipynb       # Dataset exploration
│   └── predict_using_alzehmierCNN.ipynb  # Prediction demonstration
├── outputs                      # Performance metrics
├── preentation                  # Project presentation
│   └── Presentation_TFM.pdf     # Presentation PDF
├── sample testing images        # Sample images
├── WGAN                         # WGAN resources
│   ├── WGAN_Synthetic_Images    # Generated synthetic images
│   ├── critic_epoch_100.h5      # Discriminator model
│   ├── Dataset_Sintetico.ipynb  # Image generation
│   ├── generator_epoch_100.h5   # Generator model
│   └── WGAN_Alzheimer.ipynb     # Code to create WGAN
├── .env                         # API key (Gemini)
├── app.py                       # Streamlit application
├── chatbot.py                   # Chatbot implementation
├── paciente.py                  # Patient data management
├── model_arch.py                # Model architecture
├── README.md                    # Documentation
└── requirements.txt             # Dependencies
```

## ⚠️ Disclaimer

The resources used in this project for diagnosis as well as for information about treatments or lifestyle advice **do not in any case substitute a medical consultation**.

It is recommended to use this tool only as **guidance and always supported by the judgment of a medical professional**.

## 🙏 Credits

- **Dataset extracted from Kaggle:** [kaggle.com](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data)
- **Academic article:** ***Addressing Data Scarcity and Class Imbalance in Alzheimer's Using WGANs-GP***: [Article](https://www.ijert.org/thesis-volume-12-2023)
- **TinyVGG16 Architecture**: [TinyVG](https://poloclub.github.io/cnn-explainer/)
- **Open-access brain MRI image bank**: [OASIS project](https://sites.wustl.edu/oasisbrains/)
- **Developed with Streamlit**: [streamlit.io](https://streamlit.io)
- **Logo generation**: [canva.com](https://www.canva.com/dream-lab)
- **Presentation creation**: [slidesai.io](https://www.slidesai.io/es)

## 📄 License

This project is distributed under the **MIT License**, allowing free use, modification, and distribution with proper attribution.
