from flask import Flask, request, render_template
from googletrans import Translator
from transformers import BertTokenizer, TFBertForSequenceClassification #added
from transformers import BertConfig, BertModel
import tensorflow as tf
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #added
model = TFBertForSequenceClassification.from_pretrained("C:\\Users\\manoj\\Downloads\\SAML\\SAML\\sentimentmodel")
translator = Translator()
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    text0 = request.form['text1']
    text_split = text0.split("\n")
    l=[]
    for i in range(len(text_split)):

        text_final = translator.translate(text_split[i]).text
        
        tf_batch = tokenizer([text_final],max_length=128, padding=True, truncation=True, return_tensors='tf')
        tf_outputs = model(tf_batch)
        print(tf_outputs)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        print(tf_predictions)
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        l.append([i+1,text_split[i],text_final,tf_predictions.numpy()[0,1],tf_predictions.numpy()[0,0]])
    return render_template('form.html', final=tf_predictions.numpy()[0,1], data=l)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)