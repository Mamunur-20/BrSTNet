# Import necessary libraries
import glob
import staintools
import PIL
import tqdm
import os
import cv2

# Read the target image and standardize its luminosity
target = staintools.read_image("/srv/scratch/bic/mmr/hist2t/data/STBC/HER2_luminal/BC24220/BC24220_E1.jpg")
target = staintools.LuminosityStandardizer.standardize(target)

# Initialize a stain normalizer with the 'vahadane' method and fit it to the target
stain_norm = staintools.StainNormalizer(method='vahadane')
stain_norm.fit(target)

# Get a list of all .jpg images in the data/STBC directory
images = glob.glob("data/STBC/*/*/*.jpg")

# Loop through each image
for img in tqdm.tqdm(images):
    # Read the image
    X = staintools.read_image(img)
    # Normalize the stain of the image
    X = stain_norm.transform(X)
    # Convert the image to a PIL Image object
    X = PIL.Image.fromarray(X.astype('uint8')).convert('RGB')

    # Define the path where the stained image will be saved
    path = img.replace(img.split('/')[-1],'').replace('STBC', 'image_stained')

    # Create the directory if it doesn't exist, then save the image
    os.makedirs(path, exist_ok=True)
    X.save(img.replace('STBC', 'image_stained'))

# Get a list of all .jpg images in the data/image_stained directory
path = sorted(glob.glob("data/image_stained/*/*/*.jpg")) 

# Loop through each image
for p in path:
    # Define the name of the mask image
    img_name= p.split(".")[0] + '_mask.jpg'
    # Read the image in grayscale
    img = cv2.imread(p, 0)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img,(5,5),0)
    # Apply Otsu's thresholding
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Save the mask image
    cv2.imwrite(img_name, th3)
