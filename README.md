Emotion Detection from Facial Expressions
This project explores emotion detection using facial expressions from the FER2023 dataset. The goal is to classify emotions based on images of faces using a variety of machine learning and deep learning models. These models include KNN, Random Forest, SVM, XGBoost, CNN, Decision Tree, Transfer Learning (VGG16), and DeepFace.

Key Features
Multiple Models: A variety of models are implemented and compared for emotion recognition, including classical machine learning models and advanced deep learning approaches.
Ensemble Methods: A Voting Classifier is employed to combine the strengths of multiple models.
Pre-trained Model: The DeepFace model leverages pre-trained deep learning capabilities for emotion recognition.
Transfer Learning: VGG16 is used for emotion recognition with pre-learned features.
Real-time Predictions: The project is integrated with a Gradio interface for easy deployment and interaction.
Models Used
Decision Tree: Achieved the highest accuracy of 99%.
KNN: Achieved 96% accuracy.
Random Forest: Also achieved 96% accuracy.
Voting Classifier: Combined models like XGBoost and SVM, resulting in a 95% accuracy.
XGBoost: Achieved 94% accuracy.
DeepFace: Achieved 55% accuracy, leveraging pre-trained deep learning features.
CNN: Achieved 68% accuracy with potential for further improvement.
Transfer Learning (VGG16): Achieved 53% accuracy, requiring task-specific adjustments.
SVM: Achieved 37% accuracy, showing challenges with high-dimensional data.
Project Setup
Requirements
To run the project, ensure you have the following dependencies installed:

Python 3.x
TensorFlow (for deep learning models like CNN and VGG16)
Keras (for model training and transfer learning)
OpenCV (for image processing)
Scikit-learn (for machine learning models)
Joblib (for model serialization)
DeepFace (for emotion detection using pre-trained models)
Gradio (for creating an interactive GUI)
Install the required libraries with:

bash
Copy code
pip install tensorflow keras opencv-python scikit-learn joblib deepface gradio
Running the Project
Load the Models: All models, label encoders, and preprocessing components are loaded at the start.
Image Preprocessing: Based on the selected model, preprocessing routines like resizing, normalization, and feature extraction (HOG) are applied.
Emotion Prediction: After preprocessing, the selected model predicts the emotion and outputs the result.
To launch the Gradio interface for real-time predictions:

python
Copy code
import gradio as gr

# Run the interface
interface.launch()
This will open a browser window with an interactive UI where you can upload images, select a model, and receive emotion predictions.

Model Comparison
Model	Accuracy	Key Parameters
Decision Tree	99%	Max Depth: None, Criterion: 'gini'
KNN	96%	n_neighbors: [1-20], Metric: ['euclidean', 'manhattan']
Random Forest	96%	n_estimators: [100, 500], max_depth: [None, 50]
Voting Classifier	95%	Soft Voting, Estimators: [XGBoost, SVM]
XGBoost	94%	Default Parameters for XGBClassifier
DeepFace	55%	Pre-trained model, actions: ['emotion']
CNN	68%	Conv2D Layers: 32, 64, 128 filters; Adam Optimizer
Transfer Learning	53%	VGG16 (pre-trained), Added Dense Layer (128 neurons)
SVM	37%	kernel: 'linear', probability=True
Challenges & Limitations
Overfitting: Some complex models, especially deep learning models, showed overfitting due to limited dataset size.
Dataset Quality: The FER2023 dataset's quality and variety in facial expressions could impact model accuracy.
High-Dimensional Features: Models like SVM struggled with the high-dimensional feature space of facial images.
Future Work
Model Tuning: Further hyperparameter tuning, especially for deep learning models, could improve accuracy.
Additional Preprocessing: Incorporating advanced preprocessing techniques like data augmentation may enhance model robustness.
Ensemble Methods: Further experimentation with ensemble methods could improve performance.
Transfer Learning Expansion: Fine-tuning transfer learning models like VGG16 for the specific emotion detection task can yield better results.
Conclusion
This project successfully demonstrates emotion detection from facial expressions, with a focus on comparing traditional machine learning models and deep learning techniques. The Decision Tree model achieved the best performance, but other models like KNN, Random Forest, and Voting Classifier also showed competitive results. The use of DeepFace and VGG16 Transfer Learning demonstrates the potential of leveraging pre-trained models for emotion recognition.

