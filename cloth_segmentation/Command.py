import os
import subprocess

input_folder = 'input'
output_folder = 'output/segmentation'
command = 'python process.py --image'

upper_img = []
bottom_img = []

def runprocess():
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            filepath = os.path.join(input_folder, filename)
            subprocess.run(f'{command} "{filepath}"', shell=True)
            
def add():
    for filename in os.listdir(output_folder):
        if filename.endswith('1.png'):
            upper_img.append(filename)
            print(filename + ' 상의 추가')
        elif filename.endswith('2.png'):
            bottom_img.append(filename)
            print(filename + ' 하의 추가')
        else:
            print('실패')
            
## runprocess()를 실행