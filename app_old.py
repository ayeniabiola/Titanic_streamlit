






import gradio as gr
import pandas as pd
import joblib

#load the trained model once, when the app starts
forest_model = joblib.load("models/RandomForest.joblib")

# PILLAR1: THE FUNCTION
def predict_survival(name, Pclass, Sex, Age,SibSp, PaCh, Fare):
    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'PaCh': [PaCh],
        'Fare': [Fare]
    })
       
    
    prediction = forest_model.predict(input_data)[0]
    probabilities = forest_model.predict_proba(input_data)[0]
    if prediction == 0:
        result = f'Wooo 😮 {name.title()}, you Died 😢'
    else:
        result = f'Cheers 🥂 {name.title()}, you Survived 😁'
        

    probs = {"Died": float(probabilities[0]),"Survived":float(probabilities[1])}
    return result, probs

#PILLAR2: THE INPUTS(one per function parameter, same order)
inputs = [
    gr.Textbox(label="Name", placeholder= "e.g Ayeni Abiola"),
    gr.Radio([1,2,3], label = "Passenger Class (1 = First)", value=3),
    gr.Radio([("Male",0 ), ("Female", 1)], label="Sex", value=0),
    gr.Slider(0, 80, value=30, step=1, label="Age"),
    gr.Number(label="Siblings/Spouses Aboard", value=0, precision=0),
    gr.Number(label="Parents/Children Aboard", value=0, precision=0),
    gr.Number(label="Fare paid ($)", value=30, precision=2)

]

outputs = [
    gr.Textbox(label="prediction"),
    gr.Label(label="probabilities")
]

demo = gr.Interface(
    fn=predict_survival,
    inputs=inputs,
    outputs=outputs,
    title="Titanic Survival Predictor",
    description= "Enter a passenger's details and see if our random forest",
    examples=[
        ["Musa Adamu", 3, 0, 20, 0, 0, 8],
        ["Abiola",1, 1, 17, 0, 1, 150],
    ],
)

if __name__ == "__main__":
    demo.launch()