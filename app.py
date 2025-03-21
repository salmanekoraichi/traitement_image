import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Configuration de la page principale
st.set_page_config(page_title="Application de Traitement d'Image", layout="wide")
st.title("Application de Traitement d'Image")
st.sidebar.title("Navigation")

# Définition des pages
pages = ["Accueil", "Transformations de Base", "Transformations Avancées", "Transformation de Fourier", "Autres Transformations"]
selection = st.sidebar.radio("Choisissez une page", pages)

# --------------------------- Page d'Accueil ---------------------------
if selection == "Accueil":
    st.header("Accueil")
    st.write("""
    Bienvenue dans cette application de traitement d'images. 
    Vous pouvez choisir dans le menu latéral l’opération que vous souhaitez effectuer sur une image. 
    L’application comprend plusieurs pages :
    
    - **Transformations de Base** : Conversion en niveaux de gris, rotation, redimensionnement et symétrie.
    - **Transformations Avancées** : Ajustement de la luminosité/du contraste, flou gaussien et détection de contours.
    - **Transformation de Fourier** : Visualisation du spectre fréquentiel d’une image.
    - **Autres Transformations** : Histogramme et égalisation d’histogramme.
    """)
    
# ------------------- Page des Transformations de Base -------------------
elif selection == "Transformations de Base":
    st.header("Transformations de Base")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Chargement et affichage de l'image originale
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_column_width=True)
        
        # Conversion de l'image en tableau numpy pour traitement
        img = np.array(image)
        
        st.subheader("Options de Transformation")
        
        # 1. Conversion en niveaux de gris
        if st.checkbox("Convertir en niveaux de gris"):
            # Attention : OpenCV utilise par défaut l'ordre BGR alors que PIL utilise RGB.
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            st.image(gray, caption="Image en Niveaux de Gris", use_column_width=True, clamp=True)
        
        # 2. Rotation
        angle = st.slider("Rotation (degrés)", -180, 180, 0)
        if angle != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            st.image(rotated, caption=f"Image Rotée de {angle}°", use_column_width=True)
        
        # 3. Redimensionnement
        scale = st.slider("Redimensionner (facteur)", 0.1, 2.0, 1.0)
        if scale != 1.0:
            resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            st.image(resized, caption=f"Image Redimensionnée (facteur {scale})", use_column_width=True)
        
        # 4. Miroir horizontal et vertical
        if st.checkbox("Miroir Horizontal"):
            flipped_h = cv2.flip(img, 1)
            st.image(flipped_h, caption="Image Miroir Horizontal", use_column_width=True)
        if st.checkbox("Miroir Vertical"):
            flipped_v = cv2.flip(img, 0)
            st.image(flipped_v, caption="Image Miroir Vertical", use_column_width=True)
            
# ----------------- Page des Transformations Avancées -----------------
elif selection == "Transformations Avancées":
    st.header("Transformations Avancées")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="advanced")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_column_width=True)
        img = np.array(image)
        
        st.subheader("Ajustement de Luminosité et Contraste")
        # Ajustement du contraste (alpha) et de la luminosité (beta)
        alpha = st.slider("Contraste (alpha)", 0.5, 3.0, 1.0)
        beta = st.slider("Luminosité (beta)", 0, 100, 0)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        st.image(adjusted, caption="Image Ajustée", use_column_width=True)
        
        st.subheader("Flou Gaussien")
        kernel = st.slider("Taille du noyau (impair)", 1, 21, 1, step=2)
        if kernel > 1:
            blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
            st.image(blurred, caption=f"Image Floutée (noyau {kernel})", use_column_width=True)
        
        st.subheader("Détection de Contours (Canny)")
        lower = st.slider("Seuil Bas", 0, 255, 100)
        upper = st.slider("Seuil Haut", 0, 255, 200)
        edges = cv2.Canny(img, lower, upper)
        st.image(edges, caption="Contours Détectés", use_column_width=True)
        
# ------------- Page de Transformation de Fourier (FFT) -------------
elif selection == "Transformation de Fourier":
    st.header("Transformation de Fourier")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="fourier")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_column_width=True)
        
        # Pour la FFT, il est préférable d'utiliser une image en niveaux de gris
        image_gray = np.array(image.convert('L'))
        
        # Calcul de la FFT et recentrage du spectre
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 pour éviter log(0)
        
        # Affichage du spectre avec matplotlib
        fig, ax = plt.subplots()
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title("Spectre de Fourier")
        ax.axis('off')
        st.pyplot(fig)
        
# ------------------- Page Autres Transformations -------------------
elif selection == "Autres Transformations":
    st.header("Autres Transformations")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="others")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_column_width=True)
        img = np.array(image)
        
        st.subheader("Histogramme de l'Image")
        if len(img.shape) == 3:
            # Pour une image couleur, afficher l'histogramme pour chaque canal
            color = ('b', 'g', 'r')
            fig, ax = plt.subplots()
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax.plot(histr, color=col)
                ax.set_xlim([0, 256])
            ax.set_title("Histogramme de l'image")
            st.pyplot(fig)
        else:
            # Pour une image en niveaux de gris
            fig, ax = plt.subplots()
            histr = cv2.calcHist([img], [0], None, [256], [0, 256])
            ax.plot(histr, color='gray')
            ax.set_xlim([0, 256])
            ax.set_title("Histogramme de l'image")
            st.pyplot(fig)
        
        st.subheader("Égalisation d'Histogramme")
        # Pour les images couleur, on égalise le canal de luminance
        if len(img.shape) == 3:
            img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            channels = cv2.split(img_y_cr_cb)
            channels[0] = cv2.equalizeHist(channels[0])
            img_y_cr_cb_eq = cv2.merge(channels)
            equalized = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
        else:
            equalized = cv2.equalizeHist(img)
        st.image(equalized, caption="Image après Égalisation d'Histogramme", use_column_width=True)
