import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import numpy as np
from nn import model
import matplotlib.pyplot as plt

imgarr = []
my_w = tk.Tk()
my_w.geometry("450x500")
my_w.title('DT project')
my_font1=('ariel', 18, 'bold')
l1 = tk.Label(my_w,text='Number Recognition',width=30,font=my_font1) 
l1.grid(row=1,column=1,columnspan=4)
my_font1=('ariel', 10)
l2 = tk.Label(my_w,text='\nSteps:-\n\n1) Click upload files and select the image\nfrom file explorer window.\n\n2)After selecting all the files\nclick the process button.\n',width=30,font=my_font1)
l2.grid(row=2,column=1,columnspan=4)

b1 = tk.Button(my_w, text='Upload Files', 
   width=20,command = lambda:upload_file())
b1.grid(row=3,column=1,columnspan=4)

b2 = tk.Button(my_w, text='Process', 
   width=20,command = lambda:process(imgarr))
b2.grid(row=4,column=1,columnspan=4)

def upload_file():

    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')] 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1
    row=5
    for f in filename:
        img=Image.open(f) 
        img=img.resize((100,100)) 
        img1 = img.resize((28,28))
        img1 = np.array(img1)

        imgarr.append(img1)
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(my_w)
        e1.grid(row=row,column=col)
        e1.image = img 
        e1['image']=img
        if(col==3): 
            row=row+1
            col=1
        else:
            col=col+1

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def process(imgarr):
    for i,x in enumerate(imgarr):
        print(x.shape == (28,28,3))
        if x.shape == (28,28,3):
            imgarr[i] = rgb2gray(x)

    imgarr = np.array(imgarr)
    print(imgarr.shape,type(imgarr))
    predictions = model.predict(imgarr)
    predictions = np.argmax(predictions, axis=1)
    fig, axes = plt.subplots(ncols=len(imgarr), sharex=False,
                sharey=True, figsize=(20, 4))
    for i in range(len(imgarr)):
        axes[i].set_title(f"prediction: {predictions[i]}")
        axes[i].imshow(imgarr[i], cmap='gray')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()
my_w.mainloop()
