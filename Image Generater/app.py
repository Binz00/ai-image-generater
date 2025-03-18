import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("🎨 AI Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion models")

# Sidebar for model selection
st.sidebar.header("Model Settings")
model_options = {
    "Dreamlike Diffusion 1.0": "dreamlike-art/dreamlike-diffusion-1.0",
    "Stable Diffusion XL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0"
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_id = model_options[selected_model]

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.warning("Running on CPU. This will be slow! Consider using a GPU for faster generation.")

# Load model button
if st.sidebar.button("Load Model"):
    with st.spinner(f"Loading {selected_model}... This may take a few minutes."):
        try:
            @st.cache_resource
            def load_model(model_id):
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                pipe = pipe.to(device)
                return pipe


            pipe = load_model(model_id)
            st.sidebar.success(f"Model {selected_model} loaded successfully!")
            st.session_state.model_loaded = True
            st.session_state.pipe = pipe
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            st.session_state.model_loaded = False

# Generation parameters
st.sidebar.header("Generation Parameters")
num_inference_steps = st.sidebar.slider("Inference Steps", 20, 150, 50)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5)
num_images = st.sidebar.slider("Number of Images", 1, 4, 1)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    use_custom_dimensions = st.checkbox("Use Custom Dimensions")
    if use_custom_dimensions:
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)
        with col2:
            width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)

    negative_prompt = st.text_area("Negative Prompt", "ugly, distorted, low quality, blurry, nsfw")

# Main content area
prompt = st.text_area("Enter your prompt:", height=100,
                      placeholder="Example: A futuristic alien with glowing blue skin and large luminous eyes floating in a luxurious swimming pool under a starry night sky...")

# Generate button
if st.button("Generate Image", type="primary", disabled=not st.session_state.get("model_loaded", False)):
    if not prompt:
        st.error("Please enter a prompt first!")
    else:
        with st.spinner("Generating your image(s)..."):
            try:
                # Prepare parameters
                params = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "negative_prompt": negative_prompt,
                    "num_images_per_prompt": num_images,
                }

                if use_custom_dimensions:
                    params["height"] = height
                    params["width"] = width

                # Generate images
                images = st.session_state.pipe(**params).images

                # Display images
                st.subheader("Generated Images")
                cols = st.columns(min(num_images, 4))

                for i, image in enumerate(images):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        st.image(image, caption=f"Image {i + 1}", use_column_width=True)

                        # Add download button for each image
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        btn = st.download_button(
                            label="Download Image",
                            data=buf.getvalue(),
                            file_name=f"generated_image_{i + 1}.png",
                            mime="image/png"
                        )

                # Store the prompt and images in session state for history
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "prompt": prompt,
                    "images": images
                })

            except Exception as e:
                st.error(f"Error generating images: {e}")

# Display generation history
if st.session_state.get("history"):
    with st.expander("Generation History"):
        for i, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Prompt {i + 1}:** {item['prompt']}")
            history_cols = st.columns(min(len(item['images']), 4))
            for j, img in enumerate(item['images']):
                col_idx = j % len(history_cols)
                with history_cols[col_idx]:
                    st.image(img, caption=f"Image {j + 1}", use_column_width=True, width=150)
            st.divider()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Diffusers")