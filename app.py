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
pages = [
    "Accueil",
    "Transformations de Base",
    "Transformations Avancées",
    "Transformation de Fourier",
    "Autres Transformations",
    "Effets Supplémentaires",
    "Filtres Spatiaux"
]
selection = st.sidebar.radio("Choisissez une page", pages)

# --------------------------- Page d'Accueil ---------------------------
if selection == "Accueil":
    st.header("Accueil")
    st.write("""
    Bienvenue dans cette application de traitement d'images. 
    Vous pouvez choisir dans le menu latéral l’opération que vous souhaitez effectuer sur une image. 
    L’application comprend plusieurs pages :
    
    - **Transformations de Base** : Conversion en niveaux de gris, rotation, redimensionnement et symétrie.
    - **Transformations Avancées** : Ajustement de luminosité/du contraste, flou gaussien et détection de contours.
    - **Transformation de Fourier** : Visualisation du spectre fréquentiel d’une image.
    - **Autres Transformations** : Histogramme et égalisation d’histogramme.
    - **Effets Supplémentaires** : Filtre sepia, pencil sketch, effet cartoon et ajout de watermark.
    - **Filtres Spatiaux** : Application de divers filtres dans le domaine spatial.
    """)

# ------------------- Page des Transformations de Base -------------------
elif selection == "Transformations de Base":
    st.header("Transformations de Base")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Chargement et affichage de l'image originale
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_container_width=True)
        
        # Conversion de l'image en tableau numpy pour traitement
        img = np.array(image)
        
        st.subheader("Options de Transformation")
        
        # 1. Conversion en niveaux de gris
        if st.checkbox("Convertir en niveaux de gris"):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            st.image(gray, caption="Image en Niveaux de Gris", use_container_width=True, clamp=True)
        
        # 2. Rotation
        angle = st.slider("Rotation (degrés)", -180, 180, 0)
        if angle != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            st.image(rotated, caption=f"Image Rotée de {angle}°", use_container_width=True)
        
        # 3. Redimensionnement
        scale = st.slider("Redimensionner (facteur)", 0.1, 2.0, 1.0)
        if scale != 1.0:
            resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            st.image(resized, caption=f"Image Redimensionnée (facteur {scale})", use_container_width=True)
        
        # 4. Miroir horizontal et vertical
        if st.checkbox("Miroir Horizontal"):
            flipped_h = cv2.flip(img, 1)
            st.image(flipped_h, caption="Image Miroir Horizontal", use_container_width=True)
        if st.checkbox("Miroir Vertical"):
            flipped_v = cv2.flip(img, 0)
            st.image(flipped_v, caption="Image Miroir Vertical", use_container_width=True)

# ----------------- Page des Transformations Avancées -----------------
elif selection == "Transformations Avancées":
    st.header("Transformations Avancées")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="advanced")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_container_width=True)
        img = np.array(image)
        
        st.subheader("Ajustement de Luminosité et Contraste")
        alpha = st.slider("Contraste (alpha)", 0.5, 3.0, 1.0)
        beta = st.slider("Luminosité (beta)", 0, 100, 0)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        st.image(adjusted, caption="Image Ajustée", use_container_width=True)
        
        st.subheader("Flou Gaussien")
        kernel = st.slider("Taille du noyau (impair)", 1, 21, 1, step=2)
        if kernel > 1:
            blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
            st.image(blurred, caption=f"Image Floutée (noyau {kernel})", use_container_width=True)
        
        st.subheader("Détection de Contours (Canny)")
        lower = st.slider("Seuil Bas", 0, 255, 100)
        upper = st.slider("Seuil Haut", 0, 255, 200)
        edges = cv2.Canny(img, lower, upper)
        st.image(edges, caption="Contours Détectés", use_container_width=True)

# ------------- Page de Transformation de Fourier -------------
elif selection == "Transformation de Fourier":
    st.header("Transformation de Fourier")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="fourier")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_container_width=True)
        image_gray = np.array(image.convert('L'))
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
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
        st.image(image, caption="Image Originale", use_container_width=True)
        img = np.array(image)
        
        st.subheader("Histogramme de l'Image")
        if len(img.shape) == 3:
            color = ('b', 'g', 'r')
            fig, ax = plt.subplots()
            for i, col in enumerate(color):
                histr = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax.plot(histr, color=col)
                ax.set_xlim([0, 256])
            ax.set_title("Histogramme de l'image")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            histr = cv2.calcHist([img], [0], None, [256], [0, 256])
            ax.plot(histr, color='gray')
            ax.set_xlim([0, 256])
            ax.set_title("Histogramme de l'image")
            st.pyplot(fig)
        
        st.subheader("Égalisation d'Histogramme")
        if len(img.shape) == 3:
            img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            channels = list(cv2.split(img_y_cr_cb))  # Conversion du tuple en liste pour modification
            channels[0] = cv2.equalizeHist(channels[0])
            img_y_cr_cb_eq = cv2.merge(channels)
            equalized = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
        else:
            equalized = cv2.equalizeHist(img)
        st.image(equalized, caption="Image après Égalisation d'Histogramme", use_container_width=True)

# ------------------- Page Effets Supplémentaires -------------------
elif selection == "Effets Supplémentaires":
    st.header("Effets Supplémentaires")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="effects")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_container_width=True)
        img = np.array(image)
        
        effect = st.selectbox("Choisissez un effet", 
                              ["Aucun", "Filtre Sepia", "Pencil Sketch", "Cartoon Effect", "Ajouter Watermark"])
        
        if effect == "Filtre Sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(img, sepia_filter)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            st.image(sepia_img, caption="Image avec Filtre Sepia", use_container_width=True)
        elif effect == "Pencil Sketch":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            inv = 255 - gray
            blurred = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
            sketch = cv2.divide(gray, 255 - blurred, scale=256)
            st.image(sketch, caption="Pencil Sketch", use_container_width=True)
        elif effect == "Cartoon Effect":
            num_bilateral = st.slider("Nombre de filtrages bilatéraux", 1, 10, 5)
            filtered = img.copy()
            for i in range(num_bilateral):
                filtered = cv2.bilateralFilter(filtered, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, 
                                          cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 9, 9)
            cartoon = cv2.bitwise_and(filtered, filtered, mask=edges)
            st.image(cartoon, caption="Cartoon Effect", use_container_width=True)
        elif effect == "Ajouter Watermark":
            watermark = "Salmane Koraichi"
            watermarked_img = img.copy()
            font_scale = st.slider("Taille de la signature", 0.5, 3.0, 1.0)
            thickness = st.slider("Épaisseur du texte", 1, 5, 2)
            (h, w) = watermarked_img.shape[:2]
            text_size, _ = cv2.getTextSize(watermark, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w, text_h = text_size
            x = w - text_w - 10
            y = h - 10
            cv2.putText(watermarked_img, watermark, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            st.image(watermarked_img, caption="Image avec Watermark", use_container_width=True)
        else:
            st.write("Sélectionnez un effet pour l'appliquer à l'image.")

# ------------------- Page Filtres Spatiaux -------------------
elif selection == "Filtres Spatiaux":
    st.header("Filtres Spatiaux")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"], key="spatial")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Originale", use_container_width=True)
        img = np.array(image)
        
        spatial_filter = st.selectbox("Choisissez un filtre spatial", 
                                      ["Aucun", "Filtre Moyenneur", "Filtre Médian", "Filtre Laplacian", 
                                       "Filtre Sobel", "Filtre Scharr", "Filtre Passe-Haut"])
        if spatial_filter == "Filtre Moyenneur":
            kernel_size = st.slider("Taille du noyau", 1, 15, 3)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            filtered = cv2.filter2D(img, -1, kernel)
            st.image(filtered, caption="Filtre Moyenneur appliqué", use_container_width=True)
        elif spatial_filter == "Filtre Médian":
            kernel_size = st.slider("Taille du noyau impair", 3, 15, 3, step=2)
            filtered = cv2.medianBlur(img, kernel_size)
            st.image(filtered, caption="Filtre Médian appliqué", use_container_width=True)
        elif spatial_filter == "Filtre Laplacian":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            st.image(laplacian, caption="Filtre Laplacian appliqué", use_container_width=True)
        elif spatial_filter == "Filtre Sobel":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            st.image(sobel, caption="Filtre Sobel appliqué", use_container_width=True)
        elif spatial_filter == "Filtre Scharr":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            scharr = cv2.magnitude(scharrx, scharry)
            scharr = np.uint8(np.clip(scharr, 0, 255))
            st.image(scharr, caption="Filtre Scharr appliqué", use_container_width=True)
        elif spatial_filter == "Filtre Passe-Haut":
            kernel_size = st.slider("Taille du noyau pour le passe-bas", 3, 15, 3, step=2)
            low_pass = cv2.blur(img, (kernel_size, kernel_size))
            high_pass = cv2.subtract(img, low_pass)
            st.image(high_pass, caption="Filtre Passe-Haut appliqué", use_container_width=True)
        else:
            st.write("Sélectionnez un filtre pour l'appliquer.")

# ------------------- Footer / Signature -------------------
st.markdown("---")
st.markdown("**Salmane Koraichi**")
