import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reasoning LLM configuration (for complex reasoning tasks)
REASONING_MODEL = os.getenv("REASONING_MODEL", "o1-mini")
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL")
REASONING_API_KEY = os.getenv("REASONING_API_KEY")

# OpenAI
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")

# Anthropic (Claude)
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

# Google (Gemini)
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# Groq (Grok)
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# DeepSeek
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

# Qianwen
QIANWEN_API_KEY=os.getenv("QIANWEN_API_KEY")
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")