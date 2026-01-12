# Canny Edge Detector (From Scratch)

This project implements the Canny Edge Detection algorithm from scratch using Python and NumPy, following the full pipeline of the original algorithm without relying on OpenCV’s built-in Canny function.

## 🚀 Features
- Custom 2D convolution implementation (supports multi-channel images)
- Gaussian kernel generation for noise reduction
- Image smoothing using Gaussian filtering
- Sobel operators for image gradient computation (Ix, Iy)
- Gradient magnitude and direction calculation
- Non-maximum suppression for edge thinning
- Hysteresis thresholding for edge linking
- Comparison with OpenCV’s Canny Edge Detector

## 🛠️ Tech Stack
- Python
- NumPy
- OpenCV (for image loading and comparison only)
- Matplotlib

## 🧠 Algorithm Pipeline
1. **Noise Reduction** using Gaussian filtering  
2. **Gradient Computation** using Sobel operators  
3. **Gradient Magnitude & Direction** calculation  
4. **Non-Maximum Suppression** to thin edges  
5. **Hysteresis Thresholding** to detect strong and weak edges  

## 📊 Visualizations
- Original vs blurred image
- Gradient images (Ix, Iy)
- Gradient magnitude and direction
- Non-maximum suppression result
- Final edge map
- Comparison with OpenCV’s Canny implementation

## 📍 What This Project Demonstrates
- Strong understanding of image processing fundamentals
- Manual implementation of convolution operations
- Edge detection theory and practice
- Numerical computation with NumPy
- Visualization and debugging of CV algorithms

## 📸 Screenshots
![building](https://github.com/user-attachments/assets/d49df681-2383-4d26-80df-6b9aba4ad2eb)<img width="669" height="501" alt="Canny_Edge_Detection" src="https://github.com/user-attachments/assets/16bf56d2-f035-4c9f-8bef-46508e3e0bf6" />


## 📄 Notes
This project was developed as part of a CS447: Computer Vision - University of Cyprus (Spring 2025) assignment and focuses on understanding the internal mechanics of the Canny Edge Detection algorithm rather than using black-box functions.
