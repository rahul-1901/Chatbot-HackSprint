import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

hack_info = """
HackSprint is an all-in-one web-based platform tailored specifically to foster continuous
technical growth, collaboration, and innovation within the student community of IIT Jodhpur.
This platform serves as a centralized ecosystem where students can participate in regular
hackathons, attempt daily tech and aptitude-based challenges, and showcase their
development skills through open submissions and a competitive leaderboard system.
The core objective of HackSprint is to shift students from passive learning to an active,
hands-on experience that builds real-world skills. By offering frequent, high-quality challenges
and problem-solving opportunities, the platform will act as a launchpad for every student
interested in development, software engineering, or any technical field. Through this platform,
students will gain exposure to a wide array of technologies and problem domains, better
preparing them for internships, jobs, and startup ecosystems.
"""

class ChatBot:
    def __init__(self):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
        docs = splitter.split_text(hack_info)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(docs, embedding=self.embeddings)

        # Use environment variable for security
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.25,
            max_output_tokens=256,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

    def get_response(self, prompt):
        try:
            relevant_docs = self.vectorstore.similarity_search_with_score(prompt, k=1)
            use_context = False
            context = ""

            if relevant_docs:
                doc, score = relevant_docs[0]
                if score < 0.5:
                    use_context = True
                    context = doc.page_content

            if use_context:
                enhanced_prompt = f"""
                Context about HackSprint platform: {context}
                Question: {prompt}
                Please provide a helpful answer based on the context provided.
                """
                response = self.llm.invoke(enhanced_prompt)
            else:
                response = self.llm.invoke(prompt)

            return response.content

        except Exception as e:
            return f"Error: {str(e)}"
