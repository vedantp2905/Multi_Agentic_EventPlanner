import os
import asyncio
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from firecrawl import FirecrawlApp
from fpdf import FPDF


import json

def create_pdf(url, data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Convert data (which is a dictionary or a list of dictionaries) to a JSON string
    data_str = json.dumps(data, indent=4)
    
    pdf.multi_cell(0, 10, txt=data_str)
    
    # Use the url parameter for the filename (replace special characters)
    filename = "".join(c for c in url if c.isalnum() or c in ['-', '_']) + ".pdf"
    
    pdf.output(filename)


def generate_text(llm, question, crawl_result):

    inputs = {'question': question}
   
    writer_agent = Agent(
    role='Customer Service Specialist',
    goal='To accurately and efficiently answer customer questions',
    backstory='''
    As a seasoned Customer Service Specialist, this bot has honed its 
    skills in delivering prompt and precise solutions to customer queries.
    With a background in handling diverse customer needs across various industries,
    it ensures top-notch service with every interaction.
    ''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

    task_writer = Task(
        description=(f'''Use the crawled data to accurately and efficiently answer customer question. 
                     The crawled data is : {crawl_result}
                     The customer question is : {question}
                     The task involves analyzing user queries and generating clear, concise, and accurate responses.'''),
        agent=writer_agent,
        expected_output="""
        - A detailed and well-sourced answer to customer's question.
        - No extra information. Just answer to the question
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        """
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2,
        context={"Customer Question is ": question}
    )

    result = crew.kickoff(inputs=inputs)
    return result

st.header('Crawled Content Generator')
mod = None
with st.sidebar:
    with st.form('Gemini/OpenAI'):
        model = st.radio('Choose Your LLM', ['Gemini','OpenAI'])
        api_key = st.text_input(f'Enter your API key', type="password")
        firecrawl_api = st.text_input(f'Enter your FireCrawl API key', type="password")
        url = st.text_input('Enter your url')
        submitted = st.form_submit_button("Submit")

if api_key:
    if model == 'OpenAI':
        async def setup_OpenAI():
            loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            os.environ["OPENAI_API_KEY"] = api_key
             
            llm = ChatOpenAI(model='gpt-4-turbo',temperature=0.6, max_tokens=200,api_key=api_key)
            print("OpenAI Configured")
            return llm

        llm = asyncio.run(setup_OpenAI())
        mod = 'OpenAI'

    elif model == 'Gemini':
        async def setup_gemini():
            loop = asyncio.get_event_loop()
            if loop is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            os.environ["GOOGLE_API_KEY"] = api_key

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                verbose=True,
                temperature=0.6,
                google_api_key=api_key
             )
            print("Gemini Configured")
            return llm

        llm = asyncio.run(setup_gemini())
        mod = 'Gemini'
    
    
    app = FirecrawlApp(api_key=firecrawl_api)
    crawl_url = url
    params = {
    'pageOptions': {
        'onlyMainContent': True
    }
    }
    crawled_data =  app.crawl_url(crawl_url, params=params, wait_until_done=True)
    
    create_pdf(url, crawled_data)

    question = st.text_input("Enter your question:")

    if st.button("Generate Answer"):

        with st.spinner("Generating Answer..."):
        
           generated_content = generate_text(llm, question,crawled_data)
           st.markdown(generated_content)
