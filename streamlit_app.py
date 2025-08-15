"""
🤖 AI Research Assistant - Streamlit Frontend
A beautiful web interface for automated research paper analysis
"""

import streamlit as st
import os
import json
import requests
import arxiv
import re
import asyncio
import nest_asyncio
from datetime import datetime
from typing import Dict, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

# Fix for async event loop issues in Streamlit
nest_asyncio.apply()

# Set event loop policy for Windows
if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Page configuration
st.set_page_config(
    page_title="🤖 AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .paper-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'assistant_initialized' not in st.session_state:
    st.session_state.assistant_initialized = False

class AIResearchAssistant:
    def __init__(self):
        # Set API keys from Streamlit secrets
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
        # Initialize with proper event loop handling
        try:
            # Try to get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        except Exception as e:
            st.error(f"Error initializing AI components: {e}")
            # Fallback initialization
            self.embeddings = None
            self.llm = None
        
        self.vectorstore = None
        self.current_paper_info = None

class ArXivScoutAgent:
    def __init__(self, assistant):
        self.assistant = assistant
        self.llm = assistant.llm
    
    def find_best_paper(self, topic: str) -> Optional[Dict]:
        """Find the best paper on ArXiv for the given topic"""
        try:
            search = arxiv.Search(
                query=topic,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = {
                    "title": result.title.strip(),
                    "authors": ", ".join([author.name for author in result.authors[:3]]),
                    "summary": result.summary[:300].strip() + "...",
                    "pdf_url": result.pdf_url,
                    "published": str(result.published.date()),
                    "categories": ", ".join(result.categories)
                }
                papers.append(paper)
            
            if not papers:
                return self.get_fallback_paper(topic)
            
            selected_paper = papers[0]
            return {
                "title": selected_paper["title"],
                "authors": selected_paper["authors"],
                "url": selected_paper["pdf_url"],
                "published": selected_paper["published"],
                "categories": selected_paper["categories"],
                "reason": f"Most relevant ArXiv paper. Published: {selected_paper['published']}"
            }
            
        except Exception as e:
            st.error(f"ArXiv search error: {e}")
            return self.get_fallback_paper(topic)
    
    def get_fallback_paper(self, topic: str) -> Dict:
        """Fallback papers for common topics"""
        fallback_papers = {
            "machine learning": {
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "url": "https://arxiv.org/pdf/1706.03762.pdf",
                "published": "2017-06-12",
                "categories": "cs.CL, cs.AI",
                "reason": "Foundational transformer paper"
            },
            "artificial intelligence": {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": "Devlin et al.",
                "url": "https://arxiv.org/pdf/1810.04805.pdf",
                "published": "2018-10-11",
                "categories": "cs.CL",
                "reason": "Influential AI language model"
            },
            "finance": {
                "title": "Deep Learning for Finance: Deep Portfolios",
                "authors": "Heaton et al.",
                "url": "https://arxiv.org/pdf/1605.07230.pdf",
                "published": "2016-05-23",
                "categories": "q-fin.PM",
                "reason": "AI applications in finance"
            }
        }
        
        topic_lower = topic.lower()
        for key, paper in fallback_papers.items():
            if key in topic_lower or any(word in topic_lower for word in key.split()):
                return paper
        
        return fallback_papers["machine learning"]

class AnalystAgent:
    def __init__(self, assistant):
        self.assistant = assistant
        self.embeddings = assistant.embeddings
    
    def process_paper(self, paper_info: Dict) -> bool:
        """Process paper into knowledge base"""
        try:
            with st.spinner("📥 Downloading PDF..."):
                response = requests.get(paper_info['url'], timeout=30)
                filename = "research_paper.pdf"
                
                with open(filename, "wb") as f:
                    f.write(response.content)
            
            with st.spinner("📄 Extracting text from PDF..."):
                loader = PyPDFLoader(filename)
                documents = loader.load()
            
            with st.spinner("✂️ Splitting document into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\\n\\n", "\\n", ".", " ", ""]
                )
                splits = text_splitter.split_documents(documents)
            
            with st.spinner("🧠 Creating AI knowledge base..."):
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory="./research_knowledge_base",
                    collection_name="research_paper"
                )
                
                st.session_state.vectorstore = vectorstore
                self.assistant.vectorstore = vectorstore
                self.assistant.current_paper_info = paper_info
            
            os.remove(filename)
            return True
            
        except Exception as e:
            st.error(f"Error processing paper: {e}")
            return False

class QASystem:
    def __init__(self, assistant):
        self.assistant = assistant
        self.llm = assistant.llm
    
    def ask_question(self, question: str) -> str:
        """Ask question about the processed paper"""
        if not st.session_state.vectorstore:
            return "❌ No paper has been processed yet."
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        return result["result"]
    
    def generate_summary(self) -> Dict[str, str]:
        """Generate comprehensive paper summary"""
        if not st.session_state.vectorstore:
            return {"error": "No paper processed yet"}
        
        summary_questions = {
            "Research Objective": "What is the main research question and objective of this paper?",
            "Methodology": "What methodology and approach was used in this research?",
            "Key Findings": "What are the key findings and main results of this study?",
            "Conclusions": "What are the main conclusions and implications of this research?",
            "Future Work": "What limitations are mentioned and what future work is suggested?",
            "Applications": "What are the potential real-world applications of this research?"
        }
        
        summary = {}
        progress_bar = st.progress(0)
        
        for i, (section, question) in enumerate(summary_questions.items()):
            with st.spinner(f"Analyzing: {section}..."):
                answer = self.ask_question(question)
                summary[section] = answer
                progress_bar.progress((i + 1) / len(summary_questions))
        
        return summary

# Initialize components with error handling
@st.cache_resource
def initialize_assistant():
    try:
        assistant = AIResearchAssistant()
        if assistant.embeddings is None or assistant.llm is None:
            st.error("Failed to initialize AI components. Please check your API keys.")
            return None, None, None, None
        
        scout = ArXivScoutAgent(assistant)
        analyst = AnalystAgent(assistant)
        qa_system = QASystem(assistant)
        return assistant, scout, analyst, qa_system
    except Exception as e:
        st.error(f"Error initializing assistant: {e}")
        return None, None, None, None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Automated research paper discovery, analysis, and interactive Q&A**")
    
    # Initialize components
    assistant, scout, analyst, qa_system = initialize_assistant()
    
    # Check if initialization was successful
    if assistant is None:
        st.error("❌ Failed to initialize AI Research Assistant. Please check your setup.")
        st.stop()
    
    # Sidebar - Search History
    with st.sidebar:
        st.header("📚 Search History")
        
        if st.session_state.search_history:
            for i, search in enumerate(reversed(st.session_state.search_history[-10:])):  # Show last 10
                with st.expander(f"🔍 {search['topic'][:30]}..."):
                    st.write(f"**Date:** {search['timestamp']}")
                    st.write(f"**Paper:** {search['paper_title'][:50]}...")
                    if st.button(f"Load This Search", key=f"load_{i}"):
                        st.session_state.current_paper = search
                        st.rerun()
        else:
            st.info("No searches yet. Start by entering a research topic!")
        
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Research Topic Input
        st.header("🔍 Research Topic")
        topic = st.text_input(
            "Enter your research topic:",
            placeholder="e.g., machine learning in healthcare, quantum computing, etc.",
            help="Enter any research topic you want to explore"
        )
        
        col_search, col_change = st.columns(2)
        
        with col_search:
            search_button = st.button("🚀 Find & Analyze Paper", type="primary")
        
        with col_change:
            if st.session_state.current_paper:
                change_button = st.button("🔄 Change Topic")
                if change_button:
                    st.session_state.current_paper = None
                    st.session_state.vectorstore = None
                    st.rerun()
    
    with col2:
        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.session_state.current_paper:
            if st.button("📋 Generate Summary", type="secondary"):
                st.session_state.show_summary = True
                st.rerun()
            
            if st.button("💬 Ask Questions", type="secondary"):
                st.session_state.show_qa = True
                st.rerun()
        else:
            st.info("Process a paper first to enable quick actions")
    
    # Process search
    if search_button and topic:
        with st.spinner("🔎 Searching for the best paper..."):
            paper_info = scout.find_best_paper(topic)
        
        if paper_info:
            st.markdown('<div class="success-box">✅ Found relevant paper!</div>', unsafe_allow_html=True)
            
            # Display paper info
            st.markdown(f"""
            <div class="paper-card">
                <h3>📄 {paper_info['title']}</h3>
                <p><strong>Authors:</strong> {paper_info['authors']}</p>
                <p><strong>Published:</strong> {paper_info['published']}</p>
                <p><strong>Categories:</strong> {paper_info['categories']}</p>
                <p><strong>Reason Selected:</strong> {paper_info['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process the paper
            success = analyst.process_paper(paper_info)
            
            if success:
                st.markdown('<div class="success-box">🎉 Paper processed successfully! Knowledge base created.</div>', unsafe_allow_html=True)
                
                # Save to history
                search_entry = {
                    "topic": topic,
                    "paper_title": paper_info['title'],
                    "paper_info": paper_info,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.search_history.append(search_entry)
                st.session_state.current_paper = search_entry
                
                st.success("Ready for Q&A and summary generation!")
            else:
                st.markdown('<div class="error-box">❌ Failed to process paper. Please try again.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ Could not find suitable paper. Please try a different topic.</div>', unsafe_allow_html=True)
    
    # Display current paper info
    if st.session_state.current_paper:
        st.markdown("---")
        st.header("📄 Current Paper")
        paper = st.session_state.current_paper['paper_info']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{paper['title']}**")
            st.write(f"Authors: {paper['authors']}")
            st.write(f"Published: {paper['published']}")
        
        with col2:
            if st.button("🔗 View PDF"):
                st.markdown(f"[Open PDF]({paper['url']})")
    
    # Q&A Section
    if st.session_state.current_paper:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Predefined questions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("❓ Main Contribution"):
                answer = qa_system.ask_question("What is the main contribution of this research?")
                st.write("**Answer:**", answer)
        
        with col2:
            if st.button("🔬 Methodology"):
                answer = qa_system.ask_question("What methodology was used in this study?")
                st.write("**Answer:**", answer)
        
        with col3:
            if st.button("📊 Key Results"):
                answer = qa_system.ask_question("What are the key results and findings?")
                st.write("**Answer:**", answer)
        
        # Custom question
        custom_question = st.text_input("Ask your own question:")
        if st.button("🤔 Get Answer") and custom_question:
            with st.spinner("Thinking..."):
                answer = qa_system.ask_question(custom_question)
            st.write("**Answer:**", answer)
    
    # Summary Section
    if hasattr(st.session_state, 'show_summary') and st.session_state.show_summary:
        st.markdown("---")
        st.header("📋 Comprehensive Summary")
        
        if st.session_state.current_paper:
            with st.spinner("Generating comprehensive summary..."):
                summary = qa_system.generate_summary()
            
            if "error" not in summary:
                for section, content in summary.items():
                    with st.expander(f"📝 {section}", expanded=True):
                        st.write(content)
            else:
                st.error(summary["error"])
        
        st.session_state.show_summary = False

if __name__ == "__main__":
    main()