from django.shortcuts import render, redirect
from django.db.models import Avg, Count

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, true
import tkinter as tk
from tkinter import filedialog



# Create your views here.
from management.forms import AdminForm
from management.models import AdminModel


def loginpage(request):
    if request.method == "POST":
        firstname = request.POST.get("firstname")
        password = request.POST.get("password")
        if firstname=='admin'and password=='admin':
            return redirect("uploadpage1")

    return render(request,'management/loginpage.html')

def uploadpage1(request):
    if request.method == "POST":
        newsid =request.POST.get('newsid')
        title = request.POST.get('title')
        text = request.POST.get('text')
        label = request.POST.get('label')

        AdminModel.objects.create(newsid=newsid, title=title, text=text, label=label)

    return render(request,"management/uploadpage1.html")


def upload_dataset(request):
    root = tk.Tk()

    canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue2', relief='raised')
    canvas1.pack()

    def getCSV():
        global df

        import_file_path = filedialog.askopenfilename()
        df = pd.read_csv(import_file_path)
        print(df)
        timon = df
        engine = create_engine('mysql://root:@localhost/fakenews')
        with engine.connect() as conn, conn.begin():
            timon.to_sql('management_adminmodel', conn, if_exists='append', index=False)

    browseButton_CSV = tk.Button(text=" Import CSV File  ", command=getCSV, bg='green', fg='white',font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 150, window=browseButton_CSV)


    root.mainloop()
    return render(request,'management/upload_dataset.html')