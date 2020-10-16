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

def add_image_to_dataset(im,all_scores,outfile):
    im_name=os.path.split(im)[-1]
    #get the name of the material + write
    name = im_name.split('@')[0].replace("-","")

    if(name in all_scores['material_name']):
        outfile.write(im_name+"\t")
        #get the scores + write
        index=all_scores['material_name'].index(name)
        for att in attributes:
            outfile.write(str(float(all_scores[att][index])))
            outfile.write('\t')
        outfile.write('\n')

def main():
    """Main function."""
    args = parse_args(False)
    #load the scores
    all_scores = np.load(args.scores_path, allow_pickle=True).item()
    attributes = list(all_scores.keys())[:-1]
    print(attributes)
    #create the output file and write attribute names
    outfile=open(args.image_path+"/attributes_dataset.txt",'w')
    for att in attributes:
        outfile.write(att+"\t")
    outfile.write("\n")
    #get the list of images
    images = np.sort(glob.glob(args.image_path+"/256px_dataset/*"))
    #put havran first (test_set)
    print(len(images))
    images_havran=[im for im in images if "havran" in im]
    print (len(images_havran))
    #for each image
    for im in images_havran:
        add_image_to_dataset(im,all_scores,outfile)
    for im in images:
        if "havran" in im: continue
        add_image_to_dataset(im,all_scores,outfile)

main()
