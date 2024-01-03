import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import gdown  # Importez la bibliothèque gdown pour télécharger depuis Google Drive
from model import Classifier
import os 
from configs import config
config()
st.title("Modèle capable de prédire la présence de tumeur cérébrale sur une photo")
st.subheader("Application réalisée par le groupe 3 des étudiants de Dakar Institute of Technology")
st.markdown("Cette application utilise un modèle de deep learning pour prédire la présence d'une tumeur cérébrale sur une photo")

# Création d'un bouton de téléchargement d'image
uploaded_file = st.file_uploader("Télécharger une image", type=['jpg', 'png', 'jpeg'])

# Vérifier si un fichier a été téléchargé
if uploaded_file is not None:
    # Afficher l'image téléchargée
    st.image(uploaded_file, use_column_width=True)

    # Définissez la même transformation que celle utilisée pour l'entraînement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

   

    # Chargez le modèle sauvegardé localement
    model = Classifier()  # Assurez-vous d'initialiser le modèle avec la même architecture
    model.load_state_dict(torch.load('model/modele_train_tumeur(1).pth')) # Charger le modèle sur le CPU
    model.eval()

    # Définissez une fonction pour effectuer des prédictions sur une nouvelle image
    def predict_image(image_path):
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # Appliquez la transformation et ajoutez une dimension pour le lot

        with torch.no_grad():
            output = model(img)

        # Appliquez la fonction softmax pour obtenir les probabilités
        probabilities = torch.softmax(output, dim=1)

        # Obtenez la classe prédite en fonction de la probabilité maximale
        _, predicted = torch.max(probabilities, 1)

        # Obtenez la probabilité associée à la classe prédite
        confidence = probabilities[0][predicted[0]].item() * 100  # En pourcentage

        if predicted.item() == 0:
            result = f"Pas de tumeur à {confidence:.1f}%."
        else:
            result = f"Presence de tumeur à {confidence:.1f}%."
        st.markdown(f"<p style='font-size: 24px; color: green;'>{result}</p>", unsafe_allow_html=True)

    # Bouton de prédiction
    if st.button("predict",):
        # Appelez la fonction predict_image avec le chemin de l'image à prédire
        result = predict_image(uploaded_file)
