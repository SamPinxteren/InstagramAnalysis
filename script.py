import cv2
import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import json
import argparse
from collections import Counter
from instagram_scraper import InstagramScraper

class InstagramAnalyser(object):
    def __init__(self, config, weights, labels):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.LABELS = labels

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

    def analyse_user(self, username, starttime=0, endtime=0, displayimage=False):
        ig = InstagramScraper(usernames=[username])
        ig.scrape(starttime, endtime)
        posts = []

        for post in ig.posts:
            p = {"handle": username}
            if 'image_file' in post:
                if post['image_file'].endswith("mp4"):
                    p['video'] = True
                else:
                    p['video'] = False
                    p.update(self.analyse_image(post['image_file'], displayimage))

            try:
                p['likes'] = post['edge_media_preview_like']['count']
            except Exception:
                p['likes'] = "NaN"

            try:
                p['comments'] = post['edge_media_to_comment']['count']
            except Exception:
                p['comments'] = "NaN"

            try:
                p['text_length'] = len(post['edge_media_to_caption']['edges'][0]['node']['text'])
            except Exception:
                p['text_length'] = "NaN"

            try:
                p['tag_amount'] = len(post['tags'])
            except Exception:
                p['tag_amount'] = "NaN"

            try:
                p['date'] = datetime.datetime.fromtimestamp(post['taken_at_timestamp'])
            except Exception:
                p['date'] = "NaN"

            cv2.waitKey(0)
            yield p

    def analyse_image(self, image_file, displayimage=False):
        image = cv2.imread(image_file)

        (H, W) = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        self.net.setInput(blob)

        layerOutputs = self.net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.7)
        object_counts = {l: 0 for l in self.LABELS}
        if len(idxs) > 0:
            object_counts.update(Counter([self.LABELS[classIDs[i]] for i in idxs.flatten()]))

        return object_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images from given Instagram handle')
    parser.add_argument('handle', help='Handle input (handle by default or file with -l flag)')
    parser.add_argument('-l', action='store_true', default=False, dest='handle_file', help='Use file with multiple handles')
    parser.add_argument('--names', default="coco.names", dest='names', help='\'names\' file used by model')
    parser.add_argument('--config', default="yolov3-spp.cfg", dest='config', help='Configuration file used by model')
    parser.add_argument('--weights', default="yolov3-spp.weights", dest='weights', help='Weights file used by model')
    parser.add_argument('--output', default="instagram_posts.csv", dest='output', help='File to write output to')
    flags = parser.parse_args()

    if flags.handle_file:
        handles = open(flags.handle).read().strip().split("\n")
        print("{} handles loaded.".format(len(handles)))
    else:
        handles = [flags.handle]
        print("Using single handle \'{}\'.".format(flags.handle))

    try:
        labels = open(flags.names).read().strip().split("\n")
    except IOError:
        print("Could not read names file \'{}\'".format(flags.names))
        raise SystemExit

    columns = ["handle", "date", "likes", "comments", "text_length", "tag_amount", "video"] + labels
    df = pd.DataFrame(columns=columns)

    for handle in handles:
        ia = InstagramAnalyser(flags.config, flags.weights, labels)
        for post in ia.analyse_user(handle):
            df = df.append(post, ignore_index=True)

    df = df.fillna(value=0).replace("NaN", 0)
    df.to_csv(flags.output, index_label="id")
