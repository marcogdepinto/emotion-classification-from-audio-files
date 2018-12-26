import os
import subprocess

# Loop into the filesystem
for root, dirs, files in os.walk("./Actor_09", topdown=False):
    # Loop through files
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            
            # Using ffmpeg to convert the mp4 in wav
            # Example command: "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
            command = "ffmpeg -i /Users/marcogdepinto/Desktop/Actor_09/"+ name + " " + "-ab 160k -ac 2 -ar 44100 -vn /Users/marcogdepinto/Desktop/Converted/" +  name[:-3] + "wav"

            # Execute conversion
            try:
                subprocess.call(command, shell=True)
                
            # Skip the file in case of error
            except ValueError:
                continue