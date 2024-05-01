import streamlit as st 
from lida import Manager, TextGenerationConfig, llm   
import os
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    
    
    return Image.open(BytesIO(byte_data))

lida = Manager(text_gen=llm("Openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, use_cache=True)

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph"])

if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        # Attempt to read the CSV file with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_uploader, encoding=encoding)
                break
            except UnicodeDecodeError:
                pass
        
        if df is not None:
            path_to_save = "filename.csv"
            df.to_csv(path_to_save, index=False)
            summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
            st.write(summary)
            goals = lida.goals(summary, n=2, textgen_config=textgen_config)
            for goal in goals:
                st.write(goal)
            i = 0
            library = "seaborn"
            textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
            charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
            img_base64_string = charts[0].raster
            img = base64_to_image(img_base64_string)
            st.image(img)
        else:
            st.error("Failed to read the CSV file. Please try again.")
        
elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        # Attempt to read the CSV file with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_uploader, encoding=encoding)
                break
            except UnicodeDecodeError:
                pass
        
        if df is not None:
            path_to_save = "filename1.csv"
            df.to_csv(path_to_save, index=False)
            text_area = st.text_area("Query your Data to Generate Graph", height=200)
            if st.button("Generate Graph"):
                if len(text_area) > 0:
                    st.info("Your Query: " + text_area)
                    lida = Manager(text_gen=llm("openai")) 
                    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                    summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                    user_query = text_area
                    charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                    charts[0]
                    image_base64 = charts[0].raster
                    img = base64_to_image(image_base64)
                    st.image(img)
        else:
            st.error("Failed to read the CSV file. Please try again.")
