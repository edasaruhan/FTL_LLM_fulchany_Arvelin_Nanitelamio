import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #00274D; /* Dark blue background color */
        color: #FFFFFF;
    }
    
    /* Center content */
    .main .block-container {
        max-width: 700px;
        margin: auto;
        padding-top: 50px;
    }
    
    /* Title styling */
    .stApp .stTitle h1 {
        color: #FFFFFF;
        font-family: 'Arial Black', sans-serif;
        font-size: 3em;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Input box styling */
    .stTextInput input {
        border: 2px solid #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        font-size: 1.2em;
        background-color: #333333;
        color: #FFFFFF;
    }
    
    /* Button styling */
    .stButton button {
        background-color: ##2803fc;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Generated text styling */
    .stApp .stMarkdown {
        color: #FFFFFF;
        font-family: 'Verdana', sans-serif;
        font-size: 1.2em;
        text-align: justify;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("🌍 SDG Text Generation Application")

# Text input for the prompt
prompt = st.text_input("Enter your prompt related to SDGs:")

# Generate button
if st.button("Generate"):
    if prompt:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Adjust generation parameters to reduce repetition
        outputs = model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            repetition_penalty=1.2,  # Penalizes repetition
            temperature=0.7,         # Adds some randomness
            top_k=50,                # Considers top 50 tokens at each step
            top_p=0.9                # Nucleus sampling, considering top 90% probability mass
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### Generated Text:")
        st.markdown(generated_text)
    else:
        st.write("Please enter a prompt.")
