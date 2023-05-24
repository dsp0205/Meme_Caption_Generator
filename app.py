import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from dall_e import map_pixels_to_id, unmap_id_to_pixel
from langchain.utils import find_closest_template
from apikey import apikey

os.environ['OPENAI_API_KEY'] = apikey



def load_and_preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

model = MobileNetV2(weights='imagenet')

def predict_meme_template(image):
    image = image.resize((256, 256))
    img_array = np.array(image)
    img_id = map_pixels_to_id(img_array)
    img_embedding = dall_e.encode(img_id)
    closest_template = find_closest_template(img_embedding, meme_embeddings)
    return closest_template

import streamlit as st

st.title('😂 Meme Caption Generator')

meme_template = st.text_input('Enter the meme template name')
uploaded_image = st.file_uploader("Or upload a meme image")

if uploaded_image:
    image = Image.open(uploaded_image)
    predicted_template = predict_meme_template(image)
    st.write(f'Predicted meme template: {predicted_template}')
    correct_template = st.text_input('If the predicted template is incorrect, please enter the correct template name')
    if correct_template:
        meme_template = correct_template

target_audience = st.text_input('Enter the target audience')
product = st.text_input('Enter the product name')
brand = st.text_input('Enter the brand name')

caption_template = PromptTemplate(
    input_variables=['meme_template', 'target_audience', 'product', 'brand'],
    template='Create a funny caption for the {meme_template} meme targeting {target_audience} for {product} by {brand}'
)

caption_memory = ConversationBufferMemory(input_key='meme_template', memory_key='chat_history')

llm = OpenAI(temperature=0.7)
caption_chain = LLMChain(llm=llm, prompt=caption_template, verbose=True, output_key='caption', memory=caption_memory)

caption = st.empty()

if meme_template:
    new_caption = caption_chain.run(meme_template, target_audience, product, brand)
    caption.write(new_caption)

    regenerate_caption = st.button("Regenerate Caption")
    if regenerate_caption:
        new_caption = caption_chain.run(meme_template, target_audience, product, brand)
        caption.write(new_caption)

        with st.expander('Caption History'):
            st.info(caption_memory.buffer)
