import os

if __name__ == "__main__":
    path = "../media/dataset/AFDB_annotation"
    save_path = '../media/dataset/AFDB_COCO_annotation'
    text_list = os.listdir(path)
    for text_file in text_list:
        image_path = os.path.join(path, text_file)
        save_image_path = os.path.join(save_path, text_file)
        with open(image_path,'r') as f:
            while True:
                line = f.readline()
                if not line: break
                with open(save_image_path,'a') as s_f:
                    data_list = line.split(' ')
                    if data_list[0] == '0':
                        data_list[0] = '1'
                        line = " ".join(data_list)
                        s_f.write(line)
                    elif data_list[0] == '1':
                        data_list[0] = '0'
                        line = " ".join(data_list)
                        s_f.write(line)
                    else:
                        data_list[0] = '1'
                        line = " ".join(data_list)
                        s_f.write(line)