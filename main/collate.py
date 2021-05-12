import numpy as np
import os
import glob

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})

def combine_annotations(anno_list):
    sorted_list = sorted(anno_list, key=lambda x: x[0])
    anno_block = np.array(sorted_list)

    return anno_block


if __name__ == "__main__":
    path = "output/YOLO_darknet/*.txt"
    num_holes = 11

    all_filenames = glob.glob(path)
    annotations_dict = {}
    for i in range(num_holes):
        annotations_dict[i] = []

    for filename in all_filenames:
        if os.path.getsize(filename) > 0:
            frame_num = filename.split("_")[-1].split(".")[0]
            with open(filename, 'r') as f:
                for line in f:
                    print(line)
                    split_line = line.split(" ")
                    annotation = np.zeros(5)
                    annotation[0] = frame_num

                    for i in range(1, 5):
                        annotation[i] = split_line[i]
                    annotations_dict[int(split_line[0])].append(annotation)



    lumped_dict = {}
    for i in range(num_holes):
        lumped_dict[i] = combine_annotations(annotations_dict[i])

    for i in range(num_holes):
        print("ID", i)
        print(lumped_dict[i])