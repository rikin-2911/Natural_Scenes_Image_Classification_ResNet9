import streamlit as st
import torch
from PIL import Image
from resnet9_model_def import ResNet9

import torchvision.transforms as tt
# Set class names here
class_names = ['BUILDINGS', 'FOREST', 'GLACIER', 'MOUNTAIN', 'SEA', 'STREET']  # Replace with your actual labels

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet9(in_channels=3, num_classes=6)  # Adjust num_classes as needed
model.load_state_dict(torch.load("Natural_Scenes_ResNet9_Classifier.pth", map_location=device))
model.eval()
model.to(device)

# Transform function (use the same as in training)
stats = ((0.4580, 0.4340, 0.4080), (0.2292, 0.2215, 0.2231))
transform = tt.Compose([
    tt.Resize((150, 150)),
    tt.CenterCrop(130),
    tt.ToTensor(),
    tt.Normalize(*stats)

])

# Title
st.title("🔍 Natural Scenes Image Classifier")
st.subheader("Using ResNet9 CNN Architecture")
st.markdown("Upload an image to classify it into one of the trained categories:")
st.markdown("- **Buildings**")
st.markdown("- **Forest**")
st.markdown("- **Glacier**")
st.markdown("- **Mountains**")
st.markdown("- **Sea**")
st.markdown("- **Street**")

# Instructions for users
st.write("### Instructions:")
st.markdown("1. Firsty, ensure you have an image of a natural scene. OR Go to [Google Images]: (https://images.google.com/) and search for any of the above categories.")
st.markdown("2. Download the image you want to classify.")
st.markdown("3. Click on Browse files.")
st.markdown("4. Select the image file you downloaded.")
st.markdown("5. Wait for the model to classify the image.")
st.markdown("6. The predicted class and confidence level will be displayed below the image.")
st.markdown("7. Keep in mind that the model is trained on specific categories, so it may not perform well on images outside of these categories.")
st.markdown("# Keep Testing and Enjoy the App! 😊")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=150, width=150)
    st.write("Classifying...")
    st.spinner("Please wait while we process the image...")

    # Preprocess and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Display results
    predicted_class = class_names[predicted.item()]
    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"🔢 Confidence: **{confidence.item()*100:.2f}%**")
    if confidence.item() < 0.5:
        st.warning("⚠️ The model is not very confident about this prediction. Please verify with other sources.")
    else:
        st.success("✅ The model is confident about this prediction!")
        st.balloons()  
    st.write("Thank you for using the ResNet9 Natural Scenes Image Classifier!")

# Add a footer
st.write("### About This App")
st.markdown("This is a simple web application that runs on ResNet9 CNN Architecture to classify Natural Scenes images." \
" This model is trained on Intel Image Classification Dataset, which contains about 17k images of Natural Scenes from the upper given classes and Achieved 90% Accuracy on Test Set." \
"")
st.markdown("---")
st.markdown("> Made with ❤️ by [Rikin Pithadia](https://neural-portfolio-galaxy.vercel.app/)")
st.markdown("> For documentation and source code, visit my GitHub repositorty: [Natural_Scenes_Image_Classification_ResNet9](https://github.com/rikin-2911/Natural_Scenes_Image_Classification_ResNet9)")
st.markdown("> If you have any questions or feedback, feel free to reach out!")
st.markdown("> LinkedIn: [Rikin Pithadia](https://www.linkedin.com/in/rikin-pithadia-20b94729b), Instagram: [rikin_2911](https://www.instagram.com/rikin_2911/?igsh=MWkwd3BoenVidGZqZQ%3D%3D#)")
st.markdown("> This app is built using Streamlit(Frontend), PyTorch, and Python(Backend).")

