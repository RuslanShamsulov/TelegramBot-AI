from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import labels as lb

# trained model that classifies images into 1000 classes
model = ResNet50()


def start(updater, context):
    updater.message.reply_text("Hello, my friend ! \n I will tell you what is in your photo \ud83d\udd2e")


def help_(updater, context):
    updater.message.reply_text("Just send a picture and I will tell you what is it \ud83d\udd2e")


def message(updater, context):
    updater.message.reply_text('Dont write anything  \n Just send a picture \ud83d\udc41 \ud83d\udc41')


def image(updater, context):
    photo = updater.message.photo[-1].get_file()
    photo.download("img.jpg")

    img = cv2.imread("img.jpg")

    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))  # do reshape for resnet model

    pred = np.argmax(model.predict(img))     # class that can be represented for our image
    print('model predicted')
    pred = lb.lbl[pred]

    print(pred)

    updater.message.reply_text(pred)        # sending prediction to user


updater = Updater('5495558331:AAHrjzXincSI1cYjhpVefJg0I7UCSG4rdYw')
dispatcher = updater.dispatcher

# Handlers
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('help', help_))
dispatcher.add_handler(MessageHandler(Filters.text, message))
dispatcher.add_handler(MessageHandler(Filters.photo, image))

updater.start_polling()
updater.idle()
