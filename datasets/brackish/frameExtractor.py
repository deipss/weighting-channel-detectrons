import os
import subprocess
import argparse
import pathlib
import ipdb

image_types = ['.png', '.jpg', '.jpeg']
video_types = ['.avi', '.mp4']

def extractFrames(main_dir):
    print(f"Extracting frames from {main_dir}")
    output_dir = '/data/ai_data/Brackish/dataset/images'

    for sub_dir, dirs, files in os.walk(main_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in video_types:
                videoFile = os.path.join(os.path.abspath(sub_dir), filename)

                if output_dir != '':
                    fileFolder = output_dir
                else:
                    fileFolder = os.path.splitext(videoFile)[0]
                pathlib.Path(os.path.abspath(fileFolder)).mkdir(exist_ok=True, parents=True)

                fileprefix = os.path.join(fileFolder, os.path.splitext(filename)[0])

                cmd_command = ["ffmpeg", "-i", videoFile, "-vf", "scale=960:540", "-sws_flags", "bicubic", "{}-%04d.png".format(fileprefix), "-hide_banner"]
                subprocess.call(cmd_command)

                # Create a .txt file with the names of all the image files in the respective folder
                dirContent = os.listdir(fileFolder)
                for fi in dirContent:
                    if os.path.splitext(fi)[1] in image_types:
                        with open("{}".format(os.path.join(fileFolder, "inputList.txt")), "a") as f:
                            f.write("{}\n".format(fi))


if __name__ == "__main__":

    for i in 'crab  fish-big  fish-school  fish-small-shrimp  jellyfish'.split(' '):
        extractFrames(os.path.join('/data/ai_data/Brackish/dataset/videos',i))
