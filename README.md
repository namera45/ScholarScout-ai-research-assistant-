# 🤖 AI Research Assistant

**Find, analyze, and chat with research papers automatically!**

A simple tool that finds research papers on any topic, processes them, and lets you ask questions about the content.

## ✨ What it does

1. **🔍 Find Papers**: Enter any topic → AI finds the best research paper
2. **📄 Process Paper**: Downloads and reads the PDF automatically
3. **💬 Ask Questions**: Chat with the paper content using AI
4. **📋 Get Summary**: Generate comprehensive paper analysis

![Demo](https://mhjsleappy5zmnrnevdap3w.streamlit.app/)

## � Quick Start

### 1. Get the code

```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Get your API key

- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a free API key
- Copy it

### 4. Add your API key

Open `streamlit_app.py` and replace `"your_api_key_here"` with your actual key:

```python
os.environ["GOOGLE_API_KEY"] = "your_actual_api_key_here"
```

### 5. Run the app

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How to use

1. **Enter a topic** (e.g., "machine learning in healthcare")
2. **Click "Find & Analyze Paper"**
3. **Wait for processing** (30-60 seconds)
4. **Ask questions** about the paper
5. **Generate summary** for full analysis

## 📁 What's included

```
ai-research-assistant/
├── streamlit_app.py          # Main web app
├── requirements.txt          # Required packages
├── README.md                # This file
├── LICENSE                  # MIT license
└── .gitignore              # Git ignore rules
```

## 🛠️ Built with

- **Streamlit** - Web interface
- **Google Gemini** - AI for understanding papers
- **LangChain** - AI framework
- **ChromaDB** - Stores paper content
- **ArXiv** - Source for research papers

## 💡 Example topics to try

- "artificial intelligence in finance"
- "quantum computing"
- "climate change solutions"
- "medical AI applications"
- "renewable energy technology"

## ❓ Common issues

**"No current event loop" error**

- This is fixed automatically in the code

**"API key error"**

- Make sure you added your Google AI API key correctly
- Check if you have API quota remaining

**"No papers found"**

- Try a different topic
- The system will use backup papers if needed

## 🤝 Want to contribute?

1. Fork the repository
2. Make your changes
3. Submit a pull request

Ideas for improvements:

- Add more paper sources (IEEE, PubMed)
- Better UI design
- Export features
- Mobile optimization

## 📄 License

MIT License - feel free to use this project however you want!

## 🙏 Credits

- **LangChain** for the AI framework
- **Google** for Gemini AI
- **Streamlit** for the web framework
- **ArXiv** for free research papers

---

**⭐ If this helped you, please give it a star!**

**Questions? Open an issue or discussion!**
