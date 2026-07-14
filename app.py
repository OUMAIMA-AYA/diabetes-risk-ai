from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Bienvenue dans mon projet IA de prédiction du diabète !"

if __name__ == "__main__":
    app.run(debug=True)