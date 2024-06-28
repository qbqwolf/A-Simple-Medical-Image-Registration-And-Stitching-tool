# A Tool for Medical Image Registration And Stitching

# Start up
### Windows
 ```bash
conda create -n simregi python=3.8
conda activate simregi
git clone https://github.com/qbqwolf/A-Simple-Medical-Image-Registration-And-Stitching-tool.git
 conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 pip install -r requirement.txt
  ```
## Use s1_Usermain_regi.py to complete image registration
The image should be named in the form of FxRxCh1.png, and only the first channel is used for registration.  
After registration, the output directory of the image is divided into csv and img, with the deformation field used for registration stored in the csv folder and the image stored in the img folder.
### parameter
***
<ol>
    <li>framenum: Number of regions</li>
    <li>rnum: Number of rounds</li>
    <li>inbase\outbase: Input/output address</li>
    <li>nolist: The area without registration is generally used for areas with poor shooting effects or no content</li>
    <li>Hmode: 0 is the complete H-matrix, 1 is only translation, default to 0</li>
    <li>add: File suffix name</li>
    <li>dis: Maximum registration pixel difference</li>
</ol>  

***

## Use s2_Usermain_stitch.py to complete image stitching
The input image requires a dapi channel image named FxRxCh1.png.
The output result address contains the sub-address Rx\ch1\ and contains relative and absolute position information and result images
# Reference
Ma B, Ban X, Huang H, et al. A fast algorithm for material image sequential stitching[J]. Computational Materials Science, 2019, 158: 1-13.
