from dotenv import load_dotenv

from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("Welcome to the Formula 1 RAG!")
    result = app.invoke(input={"question": "Which driver has won the most Formula 1 championships?"})
    print(result["generation"])
