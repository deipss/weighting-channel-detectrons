import cv2
import json
import numpy as np
import os
import sys
import pandas as pd
import argparse
import pathlib

def importCategories(name_file):
    # Import category names list
    categoryLabels = dict()
    categoryNumbers = dict()

    with open(name_file) as f:
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


def imageToDictEntry(imagePath, imageId, shortPath):
    imageFile = cv2.imread(imagePath)

    image = dict()

    if imageFile is not None:
        height, width, _ = imageFile.shape

        # Image dictionary
        image = dict()
        image['id'] = imageId
        image['width'] = width
        image['height'] = height
        image['file_name'] = shortPath
        image['license'] = 0
    else:
        print("Could not read: " + imagePath)

    return image

def annotationToDictEntry(df_entry, annotationId, imageToIDDict, categoryToNumberDict):
    annotation = dict()
    annotation["iscrowd"] = 0
    annotation["id"] = annotationId
    annotation["image_id"]= imageToIDDict[df_entry["Filename"]]
    annotation["category_id"] = categoryToNumberDict[df_entry["Annotation tag"]]
    tl_x = df_entry["Upper left corner X"]
    tl_y = df_entry["Upper left corner Y"]
    br_x = df_entry["Lower right corner X"]
    br_y = df_entry["Lower right corner Y"]
    width = br_x-tl_x
    height = br_y-tl_y
    annotation["bbox"] = [tl_x, tl_y, width, height]
    annotation["area"] = width*height

    return annotation

def createCategoryList(categoryLabels):
    categories = []

    for catId, name in categoryLabels.items():
        catEntry = dict()
        catEntry['supercategory'] = 'None'
        catEntry['id'] = catId
        catEntry['name'] = name

        categories.append(catEntry)

    return categories


def AAUToCOCO(args):
    # Import category names list
    datasetSplit = args["datasetSplit"]
    outPath = args["outputPath"]
    imageFolderPath = args["imageFolder"]
    annotationPath = args["annotationCSV"]
    categoryfilePath = args["categories"]
    categoryLabels, categoryNumbers = importCategories(categoryfilePath)
    categoryList = createCategoryList(categoryLabels)

    annotations_df = pd.read_csv(annotationPath, sep=";")

    images = []
    annotations = []
    idToImageLookupTable = dict()
    imageToIdLookupTable = dict()
    imageIdCounter = 0
    annotationIdCounter = 0

    for filename in annotations_df["Filename"].unique():
        basename = os.path.basename(filename)
        imageIdCounter += 1
        image = imageToDictEntry(os.path.join(os.path.abspath(imageFolderPath),filename), imageIdCounter, basename)

        idToImageLookupTable[imageIdCounter] = basename
        imageToIdLookupTable[basename] = imageIdCounter

        images.append(image)


    for index, row in annotations_df.iterrows():
        annotationIdCounter += 1
        try:
            annotation = annotationToDictEntry(row, annotationIdCounter, imageToIdLookupTable, categoryNumbers)
        except Exception as e:
            print(repr(e))
            print("ERROR: It seems that something is wrong with the annotations dataframe.\nPlease ensure that:\n - the annotations are structured correctly\n - the annotation-file (.csv) delimiter is set to be a semi-colon ;\n - the input arguments are correct (e.g., -split and -annotationCSV should point toward the same dataset split)")
            print("Closing...")
            sys.exit()
        annotations.append(annotation)



    dataset_info = {'Description': datasetSplit,
                    "url": "",
                    "version": "0.0.2",
                    "year": 2020,
                    "Contributor": "Visual Analysis of People lab, AAU",
                    "date_created": "21-08-2020"}


    gtDict = dict(info = dataset_info,
                licenses = ['MIT'],
                images = images,
                annotations = annotations,
                categories = categoryList)

    pathlib.Path(outPath).mkdir(parents=True,exist_ok=True)

    with open(os.path.join(outPath, '{}_groundtruth.json'.format(datasetSplit)), 'w') as outfile:
        json.dump(gtDict, outfile)

    lookupDict = dict(idToImageLookupTable = idToImageLookupTable,
                    imageToIdLookupTable = imageToIdLookupTable,
                    categoryNametoId = categoryNumbers,
                    categoeryIdtoName = categoryLabels)

    with open(os.path.join(outPath, '{}_helper_dirs.json'.format(datasetSplit)), 'w') as outfile:
        json.dump(lookupDict, outfile)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Converts the ground truth annotations from AAU Bounding Box annotation format to MS COCO format")
    ap.add_argument("-imageFolder", "--imageFolder", type=str, help="Path to image folder", required=True)
    ap.add_argument("-annotationCSV", "--annotationCSV", type=str, help="Path to csv file with AAU Bounding Box annotations", required=True)
    ap.add_argument("-categories", "--categories", type=str, help="Path to file with categories", required=True)
    ap.add_argument("-outputPath", "--outputPath", type=str, help="Path to folder where the annotation files should be placed", required=True)
    ap.add_argument("-split", "--datasetSplit", type=str, help="The dataset split (e.g., test, train, valid)", required=True)
    args = vars(ap.parse_args())

    AAUToCOCO(args)
