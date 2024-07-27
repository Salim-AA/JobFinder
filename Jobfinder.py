import requests
from bs4 import BeautifulSoup
import time
import random
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import ollama

# Ollama model name
OLLAMA_MODEL = 'llama3.1:8b'

def search_linkedin_jobs(keywords, location, num_pages=3, max_jobs=10):
    base_url = "https://www.linkedin.com/jobs/search"
    all_jobs = []

    for keyword in keywords:
        for page in range(num_pages):
            params = {
                "keywords": keyword,
                "location": location,
                "start": page 
            }

            response = requests.get(base_url, params=params)
            soup = BeautifulSoup(response.content, 'html.parser')

            job_cards = soup.find_all("div", class_="base-card")

            page_job_count = min(len(job_cards), max_jobs)

            for card in job_cards[:max_jobs]:
                title = card.find("h3", class_="base-search-card__title")
                company = card.find("h4", class_="base-search-card__subtitle")
                link = card.find("a", class_="base-card__full-link")

                if title and company and link:
                    all_jobs.append({
                        "title": title.text.strip(),
                        "company": company.text.strip(),
                        "link": link.get('href'),
                        "keyword": keyword
                    })

            print(f"Completed page {page + 1} for keyword: {keyword}. Found {page_job_count} jobs on this page.")
            time.sleep(random.uniform(2, 5))

    return all_jobs

def get_job_description(job_url):
    response = requests.get(job_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    description = soup.find("div", class_="description__text")
    if description:
        return description.text.strip()
    return "Description not found"

def is_relevant_job(job_title, job_description, keywords):
    title_lower = job_title.lower()
    desc_lower = job_description.lower()
    return any(keyword.lower() in title_lower or keyword.lower() in desc_lower for keyword in keywords)

  


def search_jobs(keywords, location):
    search_keywords = [k.strip() for k in keywords.split(',')]
    relevance_keywords = search_keywords + ["large language model", "ai operations", "machine learning ops"]

    jobs = search_linkedin_jobs(search_keywords, location, num_pages=3, max_jobs=10)

    relevant_jobs = []

    for job in jobs:
        print(f"Analyzing job: {job['title']} at {job['company']}")

        job_description = get_job_description(job['link'])

        if is_relevant_job(job['title'], job_description, relevance_keywords):
            relevant_jobs.append({
                "title": job['title'],
                "company": job['company'],
                "link": job['link'],
                "description": job_description
            })

        time.sleep(random.uniform(1, 3))

    return relevant_jobs

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

def process_pdf(file):
    loader = PyPDFLoader(file.name)
    pages = loader.load()
    text = "\n".join([page.page_content for page in pages])
    return text

def setup_rag(text_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    splits = text_splitter.split_text(text_content)

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, retriever):
    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as a percentage, then list keywords missing, and finally provide your final thoughts.
"""

def process_resume(job_description, resume_content, task):
    full_content = f"Job Description: {job_description}\n\nResume Content: {resume_content}"
    retriever = setup_rag(full_content)

    if task == "Compare CV to Jobs":
        response = rag_chain(input_prompt1, retriever)
    else:  # "Percentage Match"
        response = rag_chain(input_prompt3, retriever)

    return response

def crawl_jobs(keywords, location):
    jobs = search_jobs(keywords, location)
    if not jobs:
        return "No relevant jobs found. Please try different keywords or location."

    job_list = ""
    for i, job in enumerate(jobs, 1):
        job_list += f"{i}. {job['title']}  -  {job['company']}\n"

    return job_list, jobs

def analyze_resume(jobs, resume_file, task):
    if resume_file is None:
        return "Please upload a CV file."

    try:
        resume_content = process_pdf(resume_file)
    except Exception as e:
        return f"Error processing the CV: {str(e)}"

    results = []
    for job in jobs:
        job_description = job['description']
        analysis = process_resume(job_description, resume_content, task)
        results.append(f"Job: {job['title']} at {job['company']}\n\nAnalysis:\n{analysis}\n\n{'='*50}\n")

    return "\n".join(results)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# JobFinder ")

    jobs_found = gr.State([])

    with gr.Row():
        keywords = gr.Textbox(label="Job Search Keywords (comma-separated)")
        location = gr.Dropdown(["United States", "United Kingdom", "Canada", "Morocco", "Germany", "France"], label="Location")

    crawl_button = gr.Button("Crawl for Jobs")
    job_output = gr.Textbox(label="Jobs Found - Limited number of output for demo purposes - ", lines=10)

    resume_file = gr.File(label="Upload CV (PDF)")

    with gr.Row():
        tell_me_button = gr.Button("Tell Me About the Resume")
        percentage_match_button = gr.Button("Percentage Match")

    analysis_output = gr.Textbox(label="Analysis Results", lines=20)

    crawl_button.click(
        fn=crawl_jobs,
        inputs=[keywords, location],
        outputs=[job_output, jobs_found]
    )

    tell_me_button.click(
        fn=analyze_resume,
        inputs=[jobs_found, resume_file, gr.Textbox(value="Tell Me About the Resume", visible=False)],
        outputs=analysis_output
    )

    percentage_match_button.click(
        fn=analyze_resume,
        inputs=[jobs_found, resume_file, gr.Textbox(value="Percentage Match", visible=False)],
        outputs=analysis_output
    )
    show_progress=False

demo.launch()