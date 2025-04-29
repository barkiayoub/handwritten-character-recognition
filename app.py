import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✏️",
    layout="centered"
)

# Page title and description
st.title("✏️ Handwritten Digit Recognition")
st.markdown("""
    Draw a digit (0-9) on the canvas below and click the 'Predict' button to see the model's prediction.
    
    This application uses a CNN model trained on the MNIST dataset to recognize handwritten digits.
""", )

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Load the pre-trained model (assuming it's saved as model.h5)
        model = tf.keras.models.load_model('models/digits_models/tf-cnn-model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = 255 - image

    
    # Resize to 28x28 pixels (MNIST format)
    image = cv2.resize(image, (28, 28))
    
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    
    # Reshape to match model input shape (1, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1)
    os.makedirs('images', exist_ok=True)
    cv2.imwrite('images/my_image.png', image[0, :, :, 0] * 255)  # Save the image for debugging
    return image


# Load the model
model = load_model()

# Create sidebar for settings
st.sidebar.header("Settings")

# Set canvas size
canvas_size = st.sidebar.slider("Canvas Size", 150, 400, 280)

# Set stroke width
stroke_width = st.sidebar.slider("Stroke Width", 10, 40, 20)

# Add model selection option (optional enhancement)
model_type = st.sidebar.selectbox(
    "Select Model",
    ["MNIST Model", "Custom Model"],
    index=0,
    disabled=True  # Disabled for now as we're only using one model
)

# Create drawing canvas
st.subheader("Draw a Digit (0-9)")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",  # Transparent background
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",  # White color for drawing
    background_color="#0E1117",  # Black background
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

# Create columns for buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# Predict button with prediction logic
if col1.button("Predict"):
    if canvas_result.image_data is not None:
        # Extract image data from canvas
        img_data = canvas_result.image_data
        
        # Check if canvas is empty
        if np.sum(img_data[:, :, 3]) == 0:  # Check alpha channel
            st.warning("Canvas is empty. Please draw a digit.")
        else:
            # Extract RGB image (ignoring alpha for processing)
            img = img_data[:, :, 0:3]
            
            # Convert to grayscale and invert (since MNIST has white digits on black background)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            inverted = cv2.bitwise_not(gray)
            
            # Preprocess the image for the model
            processed_img = preprocess_image(inverted)
            
            # Make prediction
            if model is not None:
                prediction = model.predict(processed_img)
                
                # Get predicted digit and confidence
                predicted_digit = np.argmax(prediction[0])
                confidence = prediction[0][predicted_digit] * 100
                
                # Display results
                st.subheader(f"Prediction: {predicted_digit}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Display probability bar chart
                st.subheader("Probability Distribution")
                st.bar_chart(prediction[0])
            else:
                st.error("Model is not loaded. Please check the model file.")
    else:
        st.warning("Canvas is empty. Please draw a digit.")

# Clear canvas button
if col2.button("Clear Canvas"):
    # This will trigger a rerun with a cleared canvas
    st.session_state["canvas"] = None
    st.experimental_rerun()

# Download prediction button (optional enhancement)
if col3.button("Download Result"):
    if canvas_result.image_data is not None and 'predicted_digit' in locals():
        # Create a PIL image with the prediction
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        
        # Offer download
        st.download_button(
            label="Download Image",
            data=img_byte_arr.getvalue(),
            file_name=f"digit_prediction_{predicted_digit}.png",
            mime="image/png"
        )
    else:
        st.warning("Make a prediction first.")

# Help section
with st.expander("How to Use"):
    st.markdown("""
    1. **Draw a digit**: Use your mouse or touch to draw a single digit (0-9) on the black canvas.
    2. **Make a prediction**: Click the "Predict" button to see what digit the model recognizes.
    3. **Clear the canvas**: Use the "Clear Canvas" button to start over.
    4. **Adjust settings**: Use the sidebar to change canvas size and stroke width.
    
    **Tips for better predictions**:
    - Draw digits that are centered and fill a good portion of the canvas
    - Use clear, well-defined strokes
    - The model works best with digits drawn similarly to the MNIST dataset style
    """)

# Footer with technical information
st.markdown("---")
st.caption("""
    This application uses a CNN model similar to LeNet-5 architecture, trained on the MNIST dataset.
    Created with Streamlit, TensorFlow, and OpenCV.
""")

# Add code for creating a simple LeNet-5 model in case the pre-trained model isn't available
with st.expander("Model Information (for developers)"):
    st.code('''
# Code to create and train a LeNet-5 model on MNIST (for reference)
def create_lenet5_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1), padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Train model (not executed in this app)
# model = create_lenet5_model()
# model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
# model.save('model.h5')
    ''')