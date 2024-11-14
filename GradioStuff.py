import gradio as gr
import joblib
import numpy as np

from PIL import Image

model = joblib.load('gender_model.pkl')


def predictGender(img):
    try:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = Image.open(img)
        img = img.resize((64, 64))
        img = np.array(img)
        if img.shape == (64, 64, 3):
            img = img.flatten()
        img = img.astype('float32') / 255.0
        img = img.reshape(1, -1)
        print(img)
        result = model.predict(img)
        probabilities = model.predict_proba(img)[0]*100
        probabilities = probabilities[result]
        print(result)
        print(result[0])
        gender = result[0]

        if gender == 0:
            return f"du er en mann med {probabilities}% sannsynlighet"
        else:
            return f"du er en dame med {probabilities}% sannsynlighet"


    except Exception as e:
        print(f"Could not process image {img}: {e}")


def get_image(input_img):
    return predictGender(input_img)


user_image = gr.Interface(get_image,gr.Image(type="numpy"), "text")


user_image.launch(share=True)



