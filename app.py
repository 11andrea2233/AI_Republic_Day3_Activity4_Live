import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide") #Page title (The title is News SUmmarizer Tool)

#This is the side bar
with st.sidebar :
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })
#End of side bar

#Options 
if 'messages' not in st.session_state:
    st.session_state_messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

if options == "Home":

   st.title('News Summarizer Tool')
   st.write("Welcome to the News Article Summarizer Tool, designed to provide you with clear, concise, and well-structured summaries of news articles. This tool is ideal for readers who want to quickly grasp the essential points of any news story without wading through lengthy articles. Whether you’re catching up on global events, diving into business updates, or following the latest political developments, this summarizer delivers all the important details in a brief, easily digestible format.")
   st.write("## What the Tool Does")
   st.write("The News Article Summarizer Tool reads and analyzes full-length news articles, extracting the most critical information and presenting it in a structured manner. It condenses lengthy pieces into concise summaries while maintaining the integrity of the original content. This enables users to quickly understand the essence of any news story.")
   st.write("## How It Works")
   st.write("The tool follows a comprehensive step-by-step process to create accurate and objective summaries:")
   st.write("*Analyze and Extract Information:* The tool carefully scans the article, identifying key elements such as the main event or issue, people involved, dates, locations, and any supporting evidence like quotes or statistics.")
   st.write("*Structure the Summary:* It organizes the extracted information into a clear, consistent format. This includes:")
   st.write("- *Headline:* A brief, engaging headline that captures the essence of the story.")
   st.write("- *Lead:* A short introduction summarizing the main event.")
   st.write("- *Significance:* An explanation of why the news matters.")
   st.write("- *Details:* A concise breakdown of the key points.")
   st.write("- *Conclusion:* A wrap-up sentence outlining future implications or developments.")
   st.write("# Why Use This Tool?")
   st.write("- *Time-Saving:* Quickly grasp the key points of any article without having to read through long pieces.")
   st.write("- *Objective and Neutral:* The tool maintains an unbiased perspective, presenting only factual information.")
   st.write("- *Structured and Consistent:* With its organized format, users can easily find the most relevant information, ensuring a comprehensive understanding of the topic at hand.")
   st.write("# Ideal Users")
   st.write("This tool is perfect for:")
   st.write("- Busy professionals who need to stay informed but have limited time.")
   st.write("- Students and researchers looking for quick, accurate summaries of current events.")
   st.write("- Media outlets that want to provide readers with quick takes on trending news.")
   st.write("Start using the News Article Summarizer Tool today to get concise and accurate insights into the news that matters most!")

#elif options == "About Us":
    #st.image("")   <<put the url for an image
  #  st.title("About Us")

elif options == "Model":
    st.title('News Summarizer Tool')
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="News : ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):
            System_Prompt = """
Role: Act as an objective news summarizer. Your task is to distill news articles into brief, informative summaries that convey essential details while maintaining complete neutrality and accuracy.
Goal: Provide summaries that are comprehensive enough for readers to understand the main points and context of the article, but concise enough to be easily digestible.
Instructions for Summarizing News Articles:
Identify Core Information:
Break down the article to capture the “5Ws and H”:
Who: Identify the primary individuals, groups, or organizations at the heart of the article.
What: Clarify the main event, action, or topic.
When: State any relevant timeframes or dates, especially if they provide important context.
Where: Include specific locations, regions, or relevant geographies.
Why: Mention any reasons or motivations provided for the event or action, focusing on factual explanations rather than speculation.
How: Briefly explain how the event unfolded, including methods or steps taken if detailed.
Prioritize Key Quotes and Statements:
Select only the most critical quotes or statements from the article, specifically those that:
Illustrate the perspective of a major party involved.
Convey the article's main findings or conclusions.
Paraphrase when possible to maintain brevity, while preserving the meaning.
Background and Context:
Include any essential background or context that will help the reader understand the significance of the article.
This might involve:
Relevant historical events.
Previous or related news that links to the current story.
General trends or patterns that add depth to the event.
Outline Results and Implications:
Identify any direct outcomes, potential impacts, or broader implications, particularly those affecting:
Public policy, economic trends, or social issues.
Relevant industries, communities, or demographics.
Mention likely future developments if covered in the article, to give readers an understanding of ongoing or unresolved issues.
Write with Clarity, Conciseness, and Neutrality:
Use clear, precise language to summarize points.
Avoid any form of subjective or speculative language unless directly quoted or stated in the article.
Stay neutral, reporting only on the information provided without adding opinion, bias, or personal interpretations.
Structure for Maximum Impact:
Single-Sentence Headline Summary: Start with one sentence that conveys the main idea or takeaway from the article.
Expanded Summary: Follow up with a 2-4 sentence detailed summary covering the specific elements mentioned (i.e., key events, context, quotes, results).
"""
            user_message = News_Article
            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})
            st.success("Insight generated successfully!")
            st.subheader("Summary : ")
            st.write(response)
            
