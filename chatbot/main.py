import numpy as np
import json
from model import create_model
import random
import tensorflow as tf
from prepare import prepare_data
import calender as cl
import pyttsx3
import speech_recognition as sr
import subprocess
import datetime
import tkinter as tk
import os
import webbrowser as wb
import threading
import weather
import wikipedia
import smtplib
import keys
from PIL import Image
from PIL import ImageTk
import time
from train import train
import matplotlib.pyplot as plt


try:
    from googlesearch import search
except:
    print("googlesearch not imported!")

with open(r"C:\Users\Kumar\Documents\SEM4\ML_AI\Project\Chatbot-cum-voice-Assistant\intents.json") as file:
    data = json.load(file)

SERVICE = cl.authenticate() 

root = tk.Tk()
root.geometry('500x500')
heading = tk.Label(root, text="ChatBot CUM Voice Assistant",
                font=('montserrat', 12, "bold"), fg="black").pack()
frame = tk.Frame(root, bg="#B2F")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
your_msg = tk.StringVar()
y_scroll_bar = tk.Scrollbar(frame)
x_scroll_bar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
msg_list = tk.Listbox(frame, height=20, width=50, yscrollcommand=y_scroll_bar.set, xscrollcommand=x_scroll_bar.set)
y_scroll_bar.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.FALSE)
x_scroll_bar.pack(side=tk.BOTTOM, fill=tk.X, expand=tk.FALSE)
msg_list.pack(side=tk.LEFT, fill=tk.BOTH)
msg_list.pack()
frame.pack()

tags = []  # Contains all the different tags
all_questions_list = []  # Contains the different question with their words tokenized
questions_tags = []  # Contains the questions tags corresponding to the questions in above list
all_question_words = []  # Contains all the words in all the questions of the dataset

pr = prepare_data(data)
all_question_words, tags, all_questions_list, questions_tags = pr.prepare(data, "intents", "all_questions", "tag")

all_questions_train = []
tags_output = []

all_questions_train, tags_output = pr.get_training_set()
all_questions_train = np.array(all_questions_train)
tags_output = np.array(tags_output)

tf.reset_default_graph()
model = create_model(all_questions_train, tags_output, tags, all_question_words)
model.fit_model(all_questions_train, tags_output)



# Preparing sub tags models
sub_tags_list = []
sub_tags_models = []

for intent in data["intents"]:
    all_words_sub_questions = []
    all_sub_tags = []
    sub_question_tags = []
    all_sub_questions_list = []

    tr = prepare_data(data)
    all_words_sub_questions, all_sub_tags, all_sub_questions_list, sub_question_tags = tr.prepare(intent, "sub_tags",
                                                                                                "questions", "sub")

    all_sub_questions_train = []
    sub_tags_output = []
    all_sub_questions_train, sub_tags_output = tr.get_training_set()
    all_sub_questions_train = np.array(all_sub_questions_train)
    sub_tags_output = np.array(sub_tags_output)

    sub_model = create_model(all_sub_questions_train, sub_tags_output, all_sub_tags, all_words_sub_questions)
    sub_model.fit_model(all_sub_questions_train, sub_tags_output)
    sub_tags_models.append(sub_model)

    sub_tags_list.extend(all_sub_tags)

tags_dict = {}
answers_dict = {}


def speak(text):
    speaker = pyttsx3.init()
    voice = speaker.getProperty('voices')
    speaker.setProperty('voice', voice[1].id)
    speaker.say(text)
    speaker.runAndWait()


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

    return said




def note(text):
    date = datetime.datetime.now()
    file_name = str(date).replace(":", "-") + "-note.txt"
    with open(file_name, "w") as f:
        f.write(text)

    subprocess.Popen(["notepad.exe", file_name])


def make_note():
    speak("What would you like me to write down? ")
    write = get_audio()
    note(write)
    speak("I've made a note of that.")
    msg_list.insert(tk.END, "Loki: I've made a note of that.")


def perform_google_search():
    speak("what would you like me to search for")
    query = get_audio()
    speak("I have the following results")
    msg_list.insert(tk.END, "Loki: I have the following results:")

    res = []
    for result in search(query, tld= "com", num=5, pause=2):
        msg_list.insert(tk.END, "Loki: " + str(result))
        res.append(result)
        wb.open(res)


def prepare_tags_list():
    for intent in data["intents"]:
        curr_tag = intent["tag"]
        s_tags_list = []
        for sub_tg in intent["sub_tags"]:
            curr_sub_tag = sub_tg["sub"]
            s_tags_list.append(curr_sub_tag)
            answers_dict[curr_sub_tag] = sub_tg["answers"]

        tags_dict[curr_tag] = s_tags_list



def wish():
    hour = int(datetime.datetime.now().hour)
    
    speak("Hey")
    time.sleep(1)
    speak("This is Loki, How can I help you")
    

def send_mails(to, body):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(keys.EMAIL, keys.PASSWORD)
    server.sendmail('4as1827000224@gmail.com', to, body)
    server.close()


prepare_tags_list()

def main():
    sentence = get_audio()
    msg_list.insert(tk.END, "You: " + sentence)
    if sentence.count("exit") > 0:
        msg_list.insert(tk.END, "Loki: Good Bye!")
        speak("Good bye")
        root.quit()

    tag = model.predict_tag(sentence)
    sub = sub_tags_models[tag].predict_tag(sentence)
    tag_word = tags[tag]

    sub_list = tags_dict.get(tag_word)
    sub_tag_word = sub_list[sub]

    if sub_tag_word == "mails-send":
        try:
            speak("Who do you want to send this mail")
            to = get_audio()
            speak("what should I say to " + to)
            body = get_audio()
            send_mails(keys.DICT[to], body)
            speak("Your mail has been sent successfully !")
            msg_list.insert(tk.END, "Loki: Your mail has been sent successfully !")
        except Exception as e:
            print(e)
            speak("Sorry, Could not send this E-mail")
            msg_list.insert(tk.END, "Loki: Sorry, Could not send this E-mail")
    elif sub_tag_word == "wikipedia-open":
        ans = answers_dict.get(sub_tag_word)
        a = random.choice(ans)
        speak(a)
        results = wikipedia.summary(sentence, sentences=2)
        speak("According to wikipedia")
        speak(results)
        msg_list.insert(tk.END, "Loki: " + str(results))
        speak("Writing it to a note. Please Wait")

        file_name = "Wikipedia_search.txt"
        with open(file_name, 'w') as f:
            f.write(results)
        subprocess.Popen(['notepad.exe', file_name])
    elif sub_tag_word == "open-spotify":
        path = keys.PATH_MUSIC
        ans = answers_dict.get(sub_tag_word)
        a = random.choice(ans)
        speak(a)
        os.startfile(path)
        msg_list.insert(tk.END, "Loki: opened Spotify")
    elif sub_tag_word == "visual-studio-code-open":
        path = keys.PATH_VS_CODE
        ans = answers_dict.get(sub_tag_word)
        a = random.choice(ans)
        speak(a)
        os.startfile(path)
        msg_list.insert(tk.END, "Loki: opened visual studio")
    elif sub_tag_word == "call-weather-api":
        speak("Please tell me the name of the city")
        print("Loki: " + "Please tell me the name of the city")
        city = get_audio()
        print("city: " + str(city))
        weather_conditions = weather.get_weather(str(city))
        speak(weather_conditions)
        speak("Writing it to a note")
        file_name = str(city)+"-Weather Conditions"+ "-note.txt"
        with open(file_name, "w") as f:
            f.write(weather_conditions)
        subprocess.Popen(["notepad.exe", file_name]) 
        msg_list.insert(tk.END, "Loki: " + str(weather_conditions))
    
    elif sub_tag_word == "make-notes":
        try:
            make_note()
        except:
            msg_list.insert(tk.END, "Loki: Try again")
            speak("try again")
    elif sub_tag_word == "search-google":
        try:
            perform_google_search()
        except:
            msg_list.insert(tk.END, "Loki: An error occurred!")
            speak("An error occurred")
    elif sub_tag_word == "know-date":
        date = cl.get_date_for_day(sentence)
        speak(date)
        msg_list.insert(tk.END, "Loki: " + str(date))

    elif sub_tag_word == "get-events":
        try:
            day = cl.get_date(sentence)
            cl.get_selected_events(SERVICE, day, msg_list, tk)
        except:
            speak("None")
            msg_list.insert(tk.END, "Loki: None")
    elif sub_tag_word == "all-events":
        try:
            cl.get_all_events(SERVICE, msg_list, tk)
        except:
            msg_list.insert(tk.END, "Loki: None")
            speak("None")
    elif sub_tag_word == "exit":
        ans = answers_dict.get(sub_tag_word)
        a = random.choice(ans)
        speak(a)
        os._exit(1)
    else:
        ans = answers_dict.get(sub_tag_word)
        a = random.choice(ans)
        speak(a)
        msg_list.insert(tk.END, "Loki: " + str(a))

    
def run():
    main_thread = threading.Thread(target=main)
    main_thread.start()


picture = tk.PhotoImage(file = keys.PATH_IMAGE)
picture = picture.subsample(5,5)

send_button = tk.Button(root, image = picture, command=run, borderwidth=0)
send_button.pack()

#start = get_audio()

#if str(start) == "Loki":
#    wish()
#    run()

wish()

root.mainloop()