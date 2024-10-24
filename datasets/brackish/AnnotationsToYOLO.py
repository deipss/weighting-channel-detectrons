import argparse
import os
import sys
import pathlib
import pandas as pd
import ipdb

def AAUToYOLO(args):
    yoloTxtFilesPath = args["yoloTxtFiles"]
    annotationPath = args["annotationCSV"]
    categoryfilePath = args["categories"]

    categoryLabels, categoryNumbers = importCategories(categoryfilePath)
    annotations_df = pd.read_csv(annotationPath, sep=";")

    yolo_df = computeYOLOBBs(annotations_df)

    for sub_dirs, dirs, files in os.walk(os.path.abspath(yoloTxtFilesPath)):
        for filename in files:
            img_name = '{}.png'.format(os.path.splitext(filename)[0])
            if img_name in yolo_df["Filename"].values:
                # Get a list of annotations from the frame and write it to the yolo txt file
                ann_list = getYOLOBB(yolo_df, img_name, categoryNumbers)
                writeToYOLOFile(filename, yoloTxtFilesPath, ann_list)
    return

def writeToYOLOFile(filename, yolopath, ann_list):
    yolo_strings = ["{} {} {} {} {}\n".format(ann[0], ann[1], ann[2], ann[3], ann[4]) for ann in ann_list]
    with open(os.path.join(yolopath,filename), 'r+') as f:
        for ystr in yolo_strings:
            if ystr not in f.read():
                f.write(ystr)

def getYOLOBB(df, img_name, categoryNumbers):
    ann_list = []
    for idx, annotation in df[df["Filename"] == img_name].iterrows():
        x = int(annotation["x"])
        y = int(annotation["y"])
        height = int(annotation["height"])
        width = int(annotation["width"])
        label = int(categoryNumbers[annotation["Annotation tag"]])

        tmp = (label, x, y, width, height)
        ann_list.append(tmp)
    return ann_list


def computeYOLOBBs(df):
    tl_x = df["Upper left corner X"]
    tl_y = df["Upper left corner Y"]
    lr_x = df["Lower right corner X"]
    lr_y = df["Lower right corner Y"]

    # Find center point, height, and width
    width = lr_x-tl_x
    height = lr_y-tl_y

    x = tl_x + width/2
    y = tl_y + height/2

    df["x"] = x
    df["y"] = y
    df["width"] = width
    df["height"] = height
    return df

def importCategories(fileName):
    # Import category names list
    categoryLabels = dict()
    categoryNumbers = dict()

    with open(fileName) as f:
        lineNumber = 1

        for line in f:
            entries = line.replace('\n', '')

            if len(entries) > 1:
                number = lineNumber
                label = entries

                categoryLabels[int(number)] = label
                categoryNumbers[label] = int(number)

            lineNumber += 1

    return (categoryLabels, categoryNumbers)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Converts the ground truth annotations from AAU Bounding Box annotation format to YOLO format")
    ap.add_argument("-imageList", "--imageList", type=str, help="Path to imagelist text file")
    ap.add_argument("-yoloTxtFiles", "--yoloTxtFiles", type=str, help="Path to YOLO txt files with annotations")
    ap.add_argument("-annotationCSV", "--annotationCSV", type=str, help="Path to csv file with AAU Bounding Box annotations")
    ap.add_argument("-categories", "--categories", type=str, help="Path to file with categories")
    args = vars(ap.parse_args())

    AAUToYOLO(args)
