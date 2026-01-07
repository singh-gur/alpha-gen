# Purpose
This is a multi agentic genAI application that helps you with investment research using data scraped from yahoo finance.

# Features:
- Given a ticker, do deep dive research on the company's financials, competitors, and recent news articles for that company. Then it should provide detailed analysis and recommendations based on the data.
- Using Yahoo finance losers list, identify and investment opportunities for the companies that are underperforming based on short term news trends but have strong fundamentals.
- Using recent news articles, identify and investment opportunities.

# Considerations:
- Build the cli app so it has different commands for different use cases.
- Use Playwright for web scraping and automation.
- Use langchain and langgraph for AI capabilities.
- use agentic rag for data retrieval and analysis.

# Tools/Stack used:
- Python
- Langgraph for agentic AI
- Observability with langfuse endpoint
- uv
- ruff
