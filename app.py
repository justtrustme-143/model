from flask import Flask, request, render_template
from simpletransformers.classification import ClassificationModel
import nest_asyncio
from threading import Thread

#Flaskın asenkron çalışması için yazıldı (birden fazla işlemi eş zamanlı isteyebiliriz)
nest_asyncio.apply()

#Flaskı Oluşturuyoruz
app = Flask(__name__)

# Modeli yüklüyoruz
model_path = 'bert_model'
model = ClassificationModel('bert', model_path, use_cuda=False)

#Etiket dönüşümü (kontrol amaçlı)
def sayidan_sonuca(sayi):
    if sayi == 0:
        return 'OTHER'
    elif sayi == 1:
        return 'SEXIST'
    elif sayi == 2:
        return 'RACIST'
    elif sayi == 3:
        return 'INSULT'
    elif sayi == 4:
        return 'PROFANITY'

#metni tahmin etmek için 
def predict(texts):
    predictions, _ = model.predict(texts)
    return [sayidan_sonuca(prediction) for prediction in predictions]

#hakaret içerip içermediğini kontrol etmek için
def check_for_insult(text):
    predictions = predict([text])
    predicted_label = predictions[0]
    print("Tahmin edilen etiket:", predicted_label)
    # Tahmin edilen etiketi kontrol etme
    if predicted_label in ['INSULT', 'RACIST', 'PROFANITY', 'SEXIST']:
        return True
    else:
        return False

#index.html e yönlendirir
@app.route('/')
def index():
    return render_template('index.php')

#tahminlere yönlendirir
@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.form['text']
    result = check_for_insult(text)
    return render_template('index.php', prediction=result, text=text)

#uygulama başlatılır
def run_app():
    app.run(port=5000)

thread = Thread(target=run_app)
thread.start()


