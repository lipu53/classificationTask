from customtkinter import *
from PIL import Image

from predict import *

app = CTk()
app.geometry("1000x800")


def selectfile():
    filename=filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file=filename
    img=Image.open(filename)
    image=CTkImage(light_image=img,dark_image=img,size=(500,500))
    imLabel=CTkLabel(app,text="",image=image)
    imLabel.place(relx = 0.5, rely = 0.5,anchor="center")



def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('../pretrained_models/model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text=predict_image(model, image_file, device='cpu')
    if(prediction_text=='bike'):
        prediction_text="This is a Bike"
    else:
        prediction_text="This is a Car"

    frame=CTkFrame(master=app,fg_color="green",border_color="black",border_width=1)
    frame.place(relx = 0.6, rely = 0.9,anchor="center")
    txt=CTkLabel(master=frame,text="",font=("Roboto",40),pady=5,padx=5)
    txt=CTkLabel(master=frame,text=prediction_text,font=("Roboto",40),pady=5,padx=5)
    txt.pack(anchor="s",expand=True,pady=3,padx=3)



button_to_select = CTkButton(master=app, text = "Choose file", fg_color = "yellow", command = selectfile)
button_to_select.pack(padx = 5, pady = 5)
button_to_select.place(relx = 0.4, rely = 0.1,anchor="center")

classify_button=CTkButton(master=app,text="Classify",fg_color="red",command=classify)
classify_button.place(relx=0.6,rely=0.1,anchor="center")

app.mainloop()


