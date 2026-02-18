😊 Human Emotions – Facial Emotion Recognition using Deep Learning
A deep learning-based system that detects and classifies human emotions from facial expressions using Convolutional Neural Networks (CNNs).

📌 Project Overview
Human emotion recognition plays a vital role in improving human-computer interaction. This project focuses on building a robust deep learning model capable of detecting facial emotions from images.

The system classifies facial expressions into the following categories:

😄 Happy

😢 Sad

😡 Angry

😲 Surprise

🤢 Disgust

😨 Fear

😐 Neutral

The model is trained and evaluated using the FERAC dataset from Kaggle and leverages transfer learning techniques with pretrained CNN architectures.

🎯 Objective
To develop an accurate facial emotion recognition system.

To compare multiple deep learning architectures.

To improve model performance using preprocessing and data augmentation techniques.

To identify the most effective architecture for emotion classification.

📂 Dataset
Dataset Used: FERAC Dataset (Kaggle)

Total Images: 770+

Image Size: Resized to 224 × 224 pixels

Categories: 7 emotion classes

⚙️ Methodology
1️⃣ Data Preprocessing
Image resizing (224 × 224)

Pixel normalization (scaling between 0 and 1)

Conversion to RGB format

Directory structuring for training and testing

2️⃣ Data Augmentation
To improve generalization and reduce overfitting:

Rotation

Horizontal flipping

Zooming

Width & height shifting

3️⃣ Deep Learning Architectures Used
We experimented with multiple pretrained models:

MobileNetV2

VGG19

ResNet50

AlexNet

After comparison, the best-performing architecture was selected for final evaluation.

🧠 Model Architecture
The system uses transfer learning:

Pretrained base model (ImageNet weights)

Frozen base layers

Custom classifier layers:

Global Average Pooling

Batch Normalization

Dropout

Dense layers

Softmax output layer

Loss Function: Categorical Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

📊 Results
Achieved competitive accuracy on the FERAC dataset.

Improved robustness using augmentation.

Successfully classified 7 emotion categories.

Demonstrated strong generalization despite class imbalance challenges.

🛠️ Technologies Used
Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Google Colab / Jupyter Notebook

🚀 Applications
Human-Computer Interaction

Behavioral Analysis

Affective Computing

Mental Health Monitoring

Smart Surveillance Systems

📌 Conclusion
This project demonstrates the effectiveness of deep convolutional neural networks in facial emotion recognition. By leveraging transfer learning and proper preprocessing strategies, the system achieves reliable emotion classification performance suitable for real-world applications.









