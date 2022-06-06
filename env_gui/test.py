from tkinter import *
import cv2
from PIL import Image, ImageTk
import time

from transformers import BeitFeatureExtractor, BeitForImageClassification
def iniciar():
    global vid1
    vid1 = cv2.VideoCapture(0)
    visualizar()

def visualizar():
    global vid1, frame1
    if vid1 is not None:
        ret, frame1 = vid1.read()
        if ret == True:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame1)
            img = ImageTk.PhotoImage(image=im)

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
            
        else:
            lblVideo.image = ""
            vid1.release()

def procesar():
    global vid1, frame1, name, predicted_class_idx
    #Actual date
    print("Almacenando imagen...")
    date = (str(time.time())).replace(".", "-")
    name = 'fruit_'+ date + '_.png'
    # Sabe img
    cv2.imwrite(name, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    
    image = Image.open(name)
    image = image.convert('RGB')
    print("Modelo de reconocimiento...")
    feature_extractor = BeitFeatureExtractor.from_pretrained('/home/ale/Documents/microsoft/beit-large-patch16-512')
    model = BeitForImageClassification.from_pretrained('/home/ale/Documents/microsoft/beit-large-patch16-512')
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_idx = model.config.id2label[predicted_class_idx]
    actualizar()
    
def actualizar():
    global name, predicted_class_idx
    print("Actualizando valores...")
    img = ImageTk.PhotoImage(Image.open(name).resize((259, 200)))
    img_column_2_id_3.config(image=img)
    img_column_2_id_3.image = img
    print("Actualizando img...")
    txt_column_0_id_2.config(text = predicted_class_idx)
    print("Actualizando txt...")
    
def finalizar():
    global vid1
    vid1.release()
    # width = btnTomadatos.winfo_width()
    # print("The width of the label is:", width, "pixels") 
    # label_grid_3.config(text = "la kuka")
    
root = Tk()
root.geometry("923x700")
root.title("Detección de frutas")

# Create widgets
# Buttons
btnIniciar = Button(root, text="Iniciar", command=iniciar)
btnFinalizar = Button(root, text="Finalizar", command=finalizar)
btnTomadatos = Button(root, text="Toma de datos", command=procesar)
# Columna 0

def font_fun(type):
    family =  "Helvetica"
    if type == 1: mytuple = (family, 20)
    if type == 2: mytuple = (family, 15)
    if type == 3: mytuple = (family, 12)
    if type == 4: mytuple = (family, 10)
    return(mytuple)
lblVideo = Label(root)
txt_column_0_id_1 = Label(root, text="Análisis:", font=font_fun(1))
txt_column_0_id_2 = Label(root, text="Resultado 1", font=font_fun(2))
txt_column_0_id_3 = Label(root, text="Resultado 2", font=font_fun(3))
txt_column_0_id_4 = Label(root, text="Resultado 3", font=font_fun(4))
# Columna 1
txt_column_1_id_2 = Label(root, text="Resultado 1", font=font_fun(2))
txt_column_1_id_3 = Label(root, text="Resultado 2", font=font_fun(3))
txt_column_1_id_4 = Label(root, text="Resultado 3", font=font_fun(4))
# Columna 2
txt_column_2_id_1 = Label(root, text="Imagen fruta", font=font_fun(3))
txt_column_2_id_2 = Label(root, text="Imagen texto", font=font_fun(3))
txt_column_2_id_3 = Label(root, text="Imagen texto", font=font_fun(2))

img = ImageTk.PhotoImage(Image.open("img.png").resize((259, 200)))
img_column_2_id_3 = Label(root, image=img)
img_column_2_id_4 = Label(root, image=img)
# ---

# General button
btnIniciar.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
btnFinalizar.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
btnTomadatos.grid(row=1, columnspan=3, padx=5, pady=5, sticky='ew')

# column 0
lblVideo.grid(rowspan=4, columnspan=2, sticky = W, padx=5, pady=2)
txt_column_0_id_1.grid(row=6, column=0, sticky = W, padx=5, pady=2)
txt_column_0_id_2.grid(row=7, column=0, sticky = W, padx=5, pady=2)
txt_column_0_id_3.grid(row=8, column=0, sticky = W, padx=5, pady=2)
txt_column_0_id_4.grid(row=9, column=0, sticky = W, padx=5, pady=2)

# Column 1
txt_column_1_id_2.grid(row=7, column=1, sticky = W, padx = 5)
txt_column_1_id_3.grid(row=8, column=1, sticky = W, padx = 5)
txt_column_1_id_4.grid(row=9, column=1, sticky = W, padx = 5)

# Column 2
txt_column_2_id_1.grid(row=2, column=2, sticky = W, padx = 5)
img_column_2_id_3.grid(row=3, column=2, sticky = W, padx = 5)
txt_column_2_id_2.grid(row=4, column=2, sticky = W, padx = 5)
img_column_2_id_4.grid(row=5, column=2, padx = 5)
txt_column_2_id_3.grid(row=6, column=2, sticky = W, padx = 5)
# txt_column_2_id_3.configure(background='black')
root.mainloop()
