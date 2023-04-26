from flask import Flask, request, jsonify, render_template
import openai
from openai.embeddings_utils import distances_from_embeddings
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "xyz**"
openai.api_key = os.getenv('OPENAI_API_KEY')

CORS(app)

conversation = ""
text = ""

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


@app.route('/', methods=["GET"])
def home():
    return render_template('base.html')


@app.route('/predict', methods=["GET", 'POST'])
def predict():
    text = request.get_json().get("message")
    response = answer_question(df, question=text)
    message = {"answer": response}
    return jsonify(message)


def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="text-davinci-003",
    question="",
    max_len=1000,
    size="ada",
    debug=False,
    max_tokens=1000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    conversation = ""

    try:
        # Create a completions using the question and context
        COMPANY = "Castleberry Media"
        prompt = f"You can say hi when the user says hi, be conversational, say thank you and be helpful to the user. Be as kind as possible.\nAnswer questions as if you worked at {COMPANY}.\n\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
       
        conversation += prompt
        response = openai.Completion.create(
            prompt=conversation,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        answer = response["choices"][0]["text"].strip()
        conversation += answer
        return answer
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
