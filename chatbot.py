import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import PyPDF2

load_dotenv('secrets.env')
api_key = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

class KnowledgeRetriever:
    def __init__(self, file_paths):
        self.documents, self.document_sources = self._load_documents(file_paths)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._build_index(self.documents)

    def _load_documents(self, file_paths):
        documents = []
        sources = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    text = f.read()
            else:
                continue
            chunks = list(chunk_text(text))
            documents.extend(chunks)
            sources.extend([file_path] * len(chunks))
        return documents, sources

    def _build_index(self, documents):
        embeddings = self.model.encode(documents)
        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i in indices[0]:
            results.append({
                "content": self.documents[i],
                "source": self.document_sources[i]
            })
        return results

class GroqChatClient:
    def __init__(self, model_id='llama3-70b-8192', system_message=None, api_key=None, retriever=None):
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = Groq()
        self.model_id = model_id
        self.retriever = retriever
        self.messages = []

        if system_message:
            self.messages.append({'role': 'system', 'content': system_message})

    def draft_message(self, prompt, role='user'):
        return {'role': role, 'content': prompt}
    
    def send_request(self, message, temperature=0, max_tokens=1000, stream=False, stop=None):
        self.messages.append(message)

        if self.retriever:
            knowledge_snippets = self.retriever.retrieve(message['content'])
            knowledge_context = "\n".join(
                [f"[Source: {res['source']}]\n{res['content']}" for res in knowledge_snippets]
            )
            message['content'] += f"\n\nRelevant Knowledge:\n{knowledge_context}"

        self.messages[0]['content'] += "\n\nPlease include the source of any information retrieved from documents in your response."

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop
        )

        if not stream:
            response = {
                'content': chat_completion.choices[0].message.content,
                'finish_reason': chat_completion.choices[0].finish_reason,
                'role': chat_completion.choices[0].message.role,
                'prompt_tokens': chat_completion.usage.prompt_tokens,
                'completion_tokens': chat_completion.usage.completion_tokens,
                'total_tokens': chat_completion.usage.total_tokens,
            }
            self.messages.append(self.draft_message(response['content'], response['role']))
            return response
        return chat_completion

if __name__ == '__main__':
    system_message = """You are a helpful and knowledgeable cybersecurity assistant with experience of over 10 years. aim to limit your responses to a minimum of 1000 tokens and be aware of context in ongoing converstations. Try to keep responses relevant and ignore input that is offensive or inappropriate. Always prioritize user satisfaction b being polite and patient.
"""
    file_paths = ['ACT_Digital_Security_Guidelines_2019.pdf', "cyber_security_for_beginners_ebook.pdf"]
    retriever = KnowledgeRetriever(file_paths)
    
    client = GroqChatClient(system_message=system_message, retriever=retriever)

    while True:
        user_input = input("Enter your message (or type 'exit', 'leave', 'stop' to end): ")
        if user_input.lower() in ('exit', 'leave', 'stop'):
            break
        
        response = client.send_request(client.draft_message(user_input), stream=False)
        print(f"\nAssistant: {response['content']}")
