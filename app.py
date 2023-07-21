import re
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate, HuggingFaceHub, LLMChain


load_dotenv()

st.title("🦜🔗 AI Name Generator")

prompt = st.text_input("Descripe your idea here:")


idea_template = PromptTemplate(
    input_variables=["idea"],
    template="""Given the following idea: "{idea}". Five possible unique names for the idea are:
    """,
)

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"max_length": 500},
)

chain = LLMChain(llm=llm, prompt=idea_template, verbose=True)


def clean(names: str) -> list[str]:
    names = re.sub(r"-", " ", names)
    names = re.sub(r'"', " ", names)
    names = names.split("\n")

    cleaned_names = []
    for name in names:
        name = name.strip()
        if name and len(name) > 3:
            cleaned_names.append(name)

    return cleaned_names


if prompt:
    names = chain.run(prompt)
    cleaned_list = clean(names)

    st.markdown(f"#### Here are your names: \n")

    for name in cleaned_list:
        st.markdown(f"- {name}")
