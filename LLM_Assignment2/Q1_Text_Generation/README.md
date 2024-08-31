To include a Table of Contents (ToC) in your README file, you can add a section at the beginning that links to different parts of the document. This helps users navigate the README more easily. Here’s how you can incorporate a ToC into the README structure:

---

# SDG Text Generation Application

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Models Used](#models-used)
   - [GPT-2](#gpt-2)
   - [Bloom](#bloom)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Code Explanation](#code-explanation)
   - [app.py](#apppy)
   - [requirements.txt](#requirements-txt)
7. [Evaluation](#evaluation)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Sample Results](#sample-results)
8. [Conclusion](#conclusion)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

## Overview

The **SDG Text Generation Application** is designed to generate relevant and coherent text based on user-provided prompts related to Sustainable Development Goals (SDGs). This application leverages state-of-the-art generative AI models to assist users in exploring and understanding various SDGs through generated text.

## Features

- **Interactive Text Generation**: Users can input prompts related to SDGs and receive AI-generated text.
- **Model Comparison**: Evaluates and compares the performance of different generative models.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and seamless user experience.
- **Performance Metrics**: Detailed evaluation of models based on fluency, relevance, diversity, readability, tone, and topic coverage.

## Models Used

### GPT-2

**GPT-2** (Generative Pre-trained Transformer 2) by OpenAI is known for its impressive text generation capabilities. It was chosen for the final implementation due to:

- **High-Quality Text Generation**: Produces coherent and contextually relevant text.
- **Flexibility and Versatility**: Handles a wide range of prompts effectively.
- **Ease of Integration**: Well-supported by Streamlit, making it straightforward to integrate.

### Bloom

**Bloom** is another advanced generative model that was evaluated as part of the project. Despite its strengths, it was not selected for the final version due to:

- **Integration Complexity**: More challenging to set up and integrate compared to GPT-2.
- **Performance Results**: While capable, GPT-2 provided more consistent results for the specific tasks.

## Installation

To set up the SDG Text Generation Application:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/sdg-text-generation-app.git
   cd sdg-text-generation-app
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install required packages using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Launch the Application**: Start the Streamlit app using the command above.
2. **Enter a Prompt**: Input a text prompt related to SDGs in the provided field.
3. **Generate Text**: Click the button to generate and view text based on the prompt.

## Code Explanation

### app.py

The core of the Streamlit application, which handles user interaction and text generation. Key functionalities include:

- **Prompt Input**: Collects user input for text generation.
- **Model Loading**: Loads the selected model (GPT-2) for generating text.
- **Text Generation**: Uses the model to generate text based on the input prompt.
- **Display Results**: Shows the generated text and related metrics to the user.

### requirements.txt

Lists all necessary Python packages for the application, including:

- `streamlit` for the web interface.
- `transformers` for the AI models.
- `torch` for model processing.

## Evaluation

### Evaluation Metrics

The models were evaluated based on:

- **Fluency**: Assessed by perplexity, indicating how well the model predicts the text.
- **Relevance**: Measured using TF-IDF and cosine similarity.
- **Diversity**: Ratio of unique words to total words.
- **Readability**: Flesch Reading Ease score.
- **Tone**: Sentiment analysis for tone classification.
- **Topic Coverage**: Analysis based on the inclusion of relevant keywords.

### Sample Results

**GPT-2:**

- **Generated Text**: [Sample text here]
- **Fluency Score**: 19.64
- **Relevance Score**: 0.39
- **Diversity Score**: 0.97
- **Length OK**: True
- **Readability Score**: 55.27
- **Tone Analysis**: Positive

**Bloom:**

- **Generated Text**: [Sample text here]
- **Fluency Score**: 14.40
- **Relevance Score**: 0.24
- **Diversity Score**: 0.97
- **Length OK**: True
- **Readability Score**: 22.55
- **Tone Analysis**: Positive

## Conclusion

GPT-2 was selected for the final implementation due to its superior integration ease, high-quality text generation, and consistent performance across various metrics. Bloom, while capable, was not chosen due to its integration complexity and less consistent results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Frontier tech leaders** for the Bootcamp and Project Idea
- **OpenAI** for GPT-2.
- **BigScience** for Bloom.
- **Hugging Face** for the Transformers library.
