# A Tool for Medical Image Registration And Stitching

## Use s1_Usermain_regi.py to complete image registration
The image should be named in the form of FxRxCh1.png, and only the first channel is used for registration.  
After registration, the output directory of the image is divided into csv and img, with the deformation field used for registration stored in the csv folder and the image stored in the img folder.

## Use s2_Usermain_stitch.py to complete image stitching
The input image requires a dapi channel image named FxRxCh1.png.
The output result address contains the sub-address Rx\ch1\ and contains relative and absolute position information and result images
