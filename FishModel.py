
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ✅ Class labels (replace with your actual fish classes)
class_names = ['animal fish', 'animal fish bass', 'black_sea_sprat', 'gilt_head_bream',
               'horse_mackerel', 'red_mullet', 'red_sea_bream', 'sea_bass',
               'shrimp', 'striped_red_mullet', 'trout']

st.set_page_config(page_title="Fish Classifier", layout="wide")
st.title("🐟 Multiclass Fish Image Classifier")
st.markdown("Upload a fish image and select a model to classify the fish!")

# ✅ Sidebar for model selection
model_name = st.sidebar.selectbox(
    "Choose a pre-trained model",
    ("vgg16_best.h5", "resnet50_best.h5", "best_mobilenet_model.h5", 
     "inception_best.h5", "efficient_best.h5","best_model.h5")
)

# ✅ Get absolute model path
model_path = os.path.join(os.getcwd(), model_name)

# ✅ Upload image
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

# ✅ Preprocess the image
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Predict and return all class scores
def predict_all_classes(preprocessed_img, model):
    preds = model.predict(preprocessed_img)[0]
    return preds

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file)

    with st.spinner(" Loading model and predicting..."):
        model = load_model(model_path)
        input_shape = model.input_shape[1:3]  # Example: (224, 224)

        # ✅ Preprocess the uploaded image
        preprocessed_img = preprocess_image(img, input_shape)

        # ✅ Prediction for all classes
        predictions = predict_all_classes(preprocessed_img, model)

        # ✅ Top prediction
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

    st.success(f"✅ **Prediction:** `{predicted_class}` ({confidence:.2f}% confidence)")

    # ✅ Bar Chart of Confidence for All Classes
    st.subheader("Confidence Scores for All Classes")
    st.bar_chart(data=predictions, use_container_width=True)

    # ✅ Show Class Labels and Confidence
    st.write("### Detailed Class Confidence:")
    for i, (cls, score) in enumerate(zip(class_names, predictions)):
        st.write(f"{i+1}. **{cls}** — `{score * 100:.2f}%`")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & TensorFlow")