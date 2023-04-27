from flask import Flask, request, jsonify, render_template
import openai
from openai.embeddings_utils import distances_from_embeddings
from flask_cors import CORS
import numpy as np
import pandas as pd
from heyoo import WhatsApp
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "xyz**"
openai.api_key = os.environ["OPENAI_API_KEY"]

ACCESS_TOKEN_TEMP = os.environ["ACCESS_TOKEN_TEMP"]
ID_PHONE_NUMBER = os.environ["ID_PHONE_NUMBER"]
TOKEN_URL = os.environ["TOKEN_URL"]
EMPRESA = os.environ["EMPRESA"]

CORS(app)

conversation = ""
text = ""

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


@app.route('/', methods=["GET"])
def home():
    return render_template('base.html')


@app.route("/webhook/", methods=["POST", "GET"])
def webhook_whatsapp():

    # SI HAY DATOS RECIBIDOS VIA GET
    if request.method == "GET":
        # SI EL TOKEN ES IGUAL AL QUE RECIBIMOS
        if request.args.get('hub.verify_token') == TOKEN_URL:
            # ESCRIBIMOS EN EL NAVEGADOR EL VALOR DEL RETO RECIBIDO DESDE FACEBOOK
            return request.args.get('hub.challenge')
        else:
            # SI NO SON IGUALES RETORNAMOS UN MENSAJE DE ERROR
            return "Error de autentificacion."
        
    # RECIBIMOS TODOS LOS DATOS ENVIADO VIA JSON
    data = request.get_json()

    # EXTRAEMOS EL NUMERO DE TELEFONO Y EL MANSAJE
    client_phone_number = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
    
    # EXTRAEMOS EL TELEFONO DEL CLIENTE
    message_body = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']

    # EXTRAEMOS EL ID DE WHATSAPP DEL ARRAY
    id_wpp = data['entry'][0]['changes'][0]['value']['messages'][0]['id']

    # EXTRAEMOS EL TIEMPO DE WHATSAPP DEL ARRAY
    timestamp_message = data['entry'][0]['changes'][0]['value']['messages'][0]['timestamp']

    # ESCRIBIMOS EL NUMERO DE TELEFONO Y EL MENSAJE
    # SI HAY UN MENSAJE
    if message_body is not None:
        #text = request.get_json().get("message")
        response = answer_question(df, question=message_body)
        message = {"answer": response}
        result = message["answer"]

        # ===========================
        #CONECTAMOS A LA BASE DE DATOS
        import mysql.connector
        mydb = mysql.connector.connect(
            host="mysql-cbmedia.alwaysdata.net",
            user="cbmedia",
            password="2eTgmcaT5pyQhf3",
            database='cbmedia_chatbots'
        )
        mycursor = mydb.cursor()
        query = "SELECT count(id) AS quatity FROM registro WHERE id_wa='" + id_wpp + "';"
        mycursor.execute(query)
        quatity, = mycursor.fetchone()
        quatity = str(quatity)
        quatity = int(quatity)
        if quatity == 0:
            sql = ("INSERT INTO registro" +
                   "(mensaje_recibido,mensaje_enviado,id_wa      ,timestamp_wa   ,telefono_cliente, empresa) VALUES " +
                   "('"+message_body+"'   ,'"+result+"','"+id_wpp+"' ,'"+timestamp_message+"','"+client_phone_number+"', '"+EMPRESA+"');")
            mycursor.execute(sql)
            mydb.commit()
        # ===========================
        responding_message(client_phone_number, result)
        # RETORNAMOS EL STATUS EN UN JSON
        return jsonify({"status": "success"}, 200)


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
    model="gpt-3.5-turbo",
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=300,
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
        prompt = f"""You are an AI assistant from {COMPANY} providing helpful advice. You have been given information about Castleberry’s products and services.
                    The user will ask you questions and demand requests. 
                    Provide a conversational answer based on the context provided.
                    You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
                    If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
                    Respond in the language that the user uses"""

        messages = [
            {"role": "assistant", "content": f"{prompt}"}
        ]
        #conversation = "Context: " + context + '\n\n --- \n\n' + "Question: " + question + "\n\n --- \n\n"
        conversation = "Context: " + context + '\n\n --- \n\n' + \
            "Question: " + question + "\n\n --- \n\n"

        messages.append({"role": "user", "content": conversation})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            frequency_penalty=0.5,
            presence_penalty=0,
            top_p=1,
            stop=stop_sequence,
        )
        answer = response["choices"][0]["message"]["content"]
        #conversation += answer
        return answer
    except Exception as e:
        print(e)
        return ""

def responding_message(receiving_phone, res):
    # TOKEN DE ACCESO DE FACEBOOK
    token = ACCESS_TOKEN_TEMP

    # IDENTIFICADOR DE NÚMERO DE TELÉFONO
    idNumeroTeléfono = ID_PHONE_NUMBER

    # INICIALIZAMOS ENVIO DE MENSAJES
    wpp_object = WhatsApp(token, idNumeroTeléfono)

    # ENVIAMOS UN MENSAJE DE TEXTO
    wpp_object.send_message(res, receiving_phone)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
