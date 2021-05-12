import numpy as np
import os
import glob

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})

"""
Group the Python lists into numpy arrays
"""
def combine_annotations(anno_list):
    sorted_list = sorted(anno_list, key=lambda x: x[0])
    anno_block = np.array(sorted_list)

    return anno_block


if __name__ == "__main__":
    path = "output/YOLO_darknet/*.txt"
    num_holes = 9

    all_filenames = glob.glob(path)
    filename_base = "_".join(os.path.basename(
            all_filenames[0]).split("_")[:-2])

    annotations_dict = {}
    for i in range(num_holes):
        annotations_dict[str(i)] = []

    for filename in all_filenames:
        if os.path.getsize(filename) > 0:
            frame_num = filename.split("_")[-1].split(".")[0]
            with open(filename, 'r') as f:
                for line in f:
                    split_line = line.split(" ")
                    annotation = np.zeros(5)
                    annotation[0] = frame_num

                    for i in range(1, 5):
                        annotation[i] = split_line[i]
                    annotations_dict[split_line[0]].append(annotation)


    lumped_dict = {}
    for i in range(num_holes):
        lumped_dict[str(i)] = combine_annotations(annotations_dict[str(i)])

    for i in range(num_holes):
        print("ID", i)
        print(lumped_dict[str(i)])

    np.savez(f"output/{filename_base}", **lumped_dict)
    print("Saved", f"output/{filename_base}.npz")