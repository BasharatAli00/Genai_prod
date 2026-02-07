from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import streamlit as st
from chatmodels.hf_model_local import chat_model   
from langchain_core.prompts import PromptTemplate, load_prompt


st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# template
template=load_prompt('template.json')

promt=template.invoke({
    'paper_input' : paper_input,
    'style_input': style_input,
    'length_input' : length_input
}
)



# promt=st.text_input("enter your promt")
if st.button("show"):
    result=chat_model.invoke(promt)

    st.write(result.content)


print("final prom twill be less")
print("final prom twill be less")