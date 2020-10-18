import os.path
import argparse
import numpy as np
import random
import glob

def parse_args(required=True):
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-i', '--image_path', type=str, required=True, default="../dataset/renders_by_geom_ldr/", 
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=required, default="all_attribute_score.npy", 
                      help='Path to the input attribute scores. (required)')
  return parser.parse_args()

def add_image_to_dataset(im,all_scores,attributes,outfile):
    im_name=os.path.split(im)[-1]
    #get the name of the material + write
    name = im_name.split('@')[0].replace("-","")
#    print(im,name)
    if(name in all_scores['material_name']):
        outfile.write(im_name+"\t")
        #get the scores + write
        index=all_scores['material_name'].index(name)
        for att in attributes:
            outfile.write(str(float(all_scores[att][index])))
            outfile.write('\t')
        outfile.write('\n')

def get_random_mat(mat_names,ratio):
    n_mat=int(len(mat_names)*ratio/100)
    print("Selectin randomly %i materials over %i"%(n_mat,len(mat_names)))
    selected_mat=random.sample(mat_names, n_mat)
    print(selected_mat)
    return selected_mat


def main():
    """Main function."""
    args = parse_args(False)
    #load the scores
    all_scores = np.load(args.scores_path, allow_pickle=True).item()
    attributes = list(all_scores.keys())[:-1]
    print(attributes)
    #create the output file and write attribute names
    outfile_test=open(args.image_path+"/attributes_dataset_test.txt",'w')
    outfile_train=open(args.image_path+"/attributes_dataset_train.txt",'w')
    for att in attributes:
        outfile_test.write(att+"\t")
        outfile_train.write(att+"\t")
    outfile_test.write("\n")
    outfile_train.write("\n")
    #get the list of images
    images = np.sort(glob.glob(args.image_path+"/256px_dataset/*"))
    print(len(images))
    #select test set
    #put havran first (test_set)
    test_list=["havran"]
    #test_list=get_random_mat(all_scores['material_name'],10)
    images_test=[im for im in images if any(test in im.replace("-","") for test in test_list)]
    print (len(images_test))
    #for each image
    for im in images_test:
        add_image_to_dataset(im,all_scores,attributes,outfile_test)
    for im in images:
        if im in images_test: continue
        add_image_to_dataset(im,all_scores,attributes,outfile_train)

main()
