from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(file_path: str):
    df = pd.read_csv(file_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = RandomForestClassifier(n_estimators=100,
                                   random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"モデルの精度: {accuracy}")
    pickle.dump(model, open("model.pkl", "wb"))

def predict_data(file_path: str):
    data = pd.read_csv(file_path)
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(data)
    return prediction

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

tools = [
    Tool(
        name="train_model",
        description="モデルを訓練する",
        func=train_model
    ),
    Tool(
        name="predict_data",
        description="データを予測する",
        func=predict_data
    )
]

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.invoke("""モデルを訓練してください。(訓練データは./iris.csvです。)
その後、予測データを出力してください。(予測データは./predict.csvです。)
""")
print(result)