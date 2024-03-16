import os
import shutil
import codecs


root_index_path = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/orderedByTools/tool_counters_dividedbysum/"
root_orderedFrame_path = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/classifiedByTools/"
root_originFrame_path = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/cholec80/frames_1fps/"

paths = sorted(os.listdir(root_index_path))

for tool_classifier in paths:
    
    print('Presenting tool_classifier', tool_classifier)
    
    index_path = root_index_path + tool_classifier
    with codecs.open(index_path, 'r') as f:
        
        if not os.path.getsize(index_path):
            print("Empty file: ", tool_classifier)
            continue
            
        folder = os.path.basename(f.name).strip(".txt")
        folder_path = os.path.join(root_orderedFrame_path, folder)
        
        if os.path.exists(folder_path):
            print("Already existed or already copied, returning...", folder)
            continue
        else:
            os.mkdir(folder_path)
            
        for line in f.readlines():
            
            if line.strip() == "":
                pass
            else:
                frame_index = line.split()[0]
                    
                origin_file_path = root_originFrame_path + folder[:-2] + '/' + frame_index + '.jpg'
                #print('origin file path: ', origin_file_path) 
                des_file_path = folder_path + '/' + frame_index + '.jpg'
                #print('des_file_path', des_file_path)
                shutil.copyfile(origin_file_path, des_file_path)


			



	
