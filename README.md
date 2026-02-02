# Underwater_Color_Disparities

### Prerequisites
1. Python 3.6
2. CUDA 9.0

### Code setup
Install all the python dependencies using pip:
        pip install -r requirements.txt

## How to use it
To run the code and enhance underwater images, simply execute:
        python UCD_main.py      

The results will be saved in the folder specified in `config.py`. You can change the input image and depth map paths in the same file.
You need to provide your own underwater images and corresponding depth maps. (You can use JointID on your dataset to generate depth maps if needed).

## File structure
- `UCD/` : Contains the main code for underwater image enhancement using color disparities.
- `UCD/config.py` : Configuration file for setting paths and parameters.
- `UCD/UCD_main.py` : Main script to run the underwater image enhancement.

## References

**Underwater_Color_Disparities**: Hao Wang et al. "Underwater Color Disparities: Cues for Enhancing Underwater Images Toward Natural Color Consistencies". In: IEEE Explore,2023. 


