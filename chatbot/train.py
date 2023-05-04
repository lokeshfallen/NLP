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


class train():

    def train():
        with open(r"C:\Users\Kumar\Documents\SEM4\ML_AI\Project\Chatbot-cum-voice-Assistant\intents.json") as file:
            data = json.load(file)
        
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

        for intent in data["intents"]:
            curr_tag = intent["tag"]
            s_tags_list = []
            for sub_tg in intent["sub_tags"]:
                curr_sub_tag = sub_tg["sub"]
                s_tags_list.append(curr_sub_tag)
                answers_dict[curr_sub_tag] = sub_tg["answers"]

            tags_dict[curr_tag] = s_tags_list

        return model, sub_tags_models, tags_dict, answers_dict