import os
import pandas as pd
import numpy as np
import json
import argparse

def VIAMEToCOCO(args):
    detections = args["detectionCSV"]
    detection_name = os.path.basename(detections)[:-4]
    threshold = args["threshold"]

    csv_df = pd.read_csv(detections, sep=",", usecols = np.arange(11))

    filenames = csv_df.iloc[:,1]
    frames = csv_df.iloc[:,2]
    TL_x = csv_df.iloc[:,3]
    TL_y = csv_df.iloc[:,4]
    BR_x = csv_df.iloc[:,5]
    BR_y = csv_df.iloc[:,6]
    detection_conf = csv_df.iloc[:,7]
    class_pred = csv_df.iloc[:,9]

    ## READ JSON HELPER DIRS INSTEAD
    with open(args["datasetName"]+'_helper_dirs.json') as f:
        helper_dirs = json.load(f)
    categoryDict = helper_dirs["categoryNametoId"]
    imageToIdDict = helper_dirs["imageToIdLookupTable"]

    coco_list = []
    for idx in range(len(csv_df)):
        if threshold > detection_conf.iloc[idx]:
            continue

        coco_list.append({"image_id":imageToIdDict[filenames.iloc[idx]],
                        "category_id":categoryDict[class_pred.iloc[idx]],
                        "bbox": [float(TL_x.iloc[idx]), float(TL_y.iloc[idx]), float(BR_x.iloc[idx]-TL_x.iloc[idx]), float(BR_y.iloc[idx]-TL_y.iloc[idx])],
                        "score":detection_conf.iloc[idx]})

    output_file = detection_name+"_" + "{:.2f}".format(threshold) +"_coco_output.json"

    with open(output_file, "w") as f:
        json.dump(coco_list, f)

    return output_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Converts the output from the VIAME detection csv to MS COCO detections")
    ap.add_argument("-detectionCSV", "--detectionCSV", type=str,
                    help="Path to the VIAME detection output csv")
    ap.add_argument("-datasetName", "--datasetName", type=str,
                    help="Desired dataset name")
    ap.add_argument("-threshold", "--threshold", type=float, default = 0.0,
                    help="Threshold of the prediction value in the VIAME detection CSV ")
    args = vars(ap.parse_args())

    VIAMEToCOCO(args)
