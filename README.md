# **NeRF-Based 3D Reconstruction and Neural View Synthesis**

## 📖 **Overview**
This project implements a **Neural Radiance Field (NeRF)** for **3D scene reconstruction** and **novel view synthesis** using **LibTorch (PyTorch C++ API)**. NeRF takes a set of 2D images of a scene along with their camera poses and generates **photorealistic 3D renderings** from unseen viewpoints by learning a continuous volumetric scene representation.

---

## **Key Features**
- **3D Scene Reconstruction**: Uses a neural network to reconstruct a 3D scene from 2D images.
- **Novel View Synthesis**: Renders new views of the scene from previously unseen camera angles.
- **Ray Marching**: Simulates light rays traveling through the 3D scene to predict color and density at each point.
- **Image Rendering**: Generates high-quality images of the reconstructed scene using volumetric rendering.

---

## 📂 **Project Structure**
```
├── include
│   ├── model.h         # NeRF Model Header
│   ├── renderer.h      # Renderer Header
│   └── utils.h         # Utility Functions Header
├── src
│   ├── model.cpp       # NeRF Model Implementation
│   ├── renderer.cpp    # Renderer Implementation
│   └── utils.cpp       # Utility Functions Implementation
├── data                # Input Data (Images, Camera Poses, Focal Length)
├── output              # Rendered Output Images
├── CMakeLists.txt      # CMake Configuration File
└── README.md           # Project Documentation (This File)
```

---

## 🛠 **Setup and Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/YourUsername/NeRF-based-3D-Reconstruction.git
cd NeRF-based-3D-Reconstruction
```

### **2. Build the Project**
Ensure you have **CMake** and a **C++ compiler** installed.
```bash
mkdir build
cd build
cmake ..
make
```

### **3. Run the Program**
```bash
./NeRF_Renderer <dataPath> <outputPath>
```
- **`<dataPath>`**: Path to the input data directory containing images, camera poses, and focal length.
- **`<outputPath>`**: Path to save the rendered output images.

---

## 📊 **Metrics for Evaluation**
Use the following metrics to evaluate the quality of rendered images:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality. Higher is better.
- **SSIM (Structural Similarity Index Measure)**: Measures structural similarity between images. Higher is better.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity. Lower is better.

---

## 🔧 **Core Components Explained**

### **1. NeRF Model (model.cpp)**
- **Input**: 3D point and viewing direction.
- **Output**: RGB color and density values for each point in the scene.
- **Positional Encoding**: Helps the model learn both small details and large structures by adding sine and cosine transformations to input coordinates.

### **2. Renderer (renderer.cpp)**
- **Ray Marching**: Shoots rays from the camera through each pixel in the image.
- **Sampling Points**: Samples points along each ray and queries the NeRF model to get color and density values.
- **Image Rendering**: Combines the sampled points to produce the final image.

### **3. Utility Functions (utils.cpp)**
- **`setSeed()`**: Sets the random seed for reproducibility.
- **`getDevice()`**: Selects the appropriate device (CPU or GPU).
- **`loadTensor()`**: Loads input data from files.
- **`saveImage()`**: Saves rendered images as PNG files.
- **`renderAndSaveOrbitViews()`**: Renders and saves multiple views from different angles to create a 360-degree video.

---

## 🎨 **Rendering Pipeline**
1. **Load Input Data**: Images, camera poses, and focal length are loaded from the input directory.
2. **Generate Rays**: Rays are generated for each pixel in the image based on the camera pose.
3. **Sample Points**: Points are sampled along each ray in the 3D scene.
4. **Predict Color and Density**: The NeRF model predicts the color and density for each sampled point.
5. **Render Image**: The predictions are combined to generate the final 2D image.
6. **Save Output**: The rendered images are saved as PNG files.

---

## 📈 **Metrics Calculation (Python Code Example)**
Use the following Python code to calculate **PSNR**, **SSIM**, and **LPIPS** between the rendered and ground truth images:
```python
import torch
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips

# PSNR Calculation
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

# SSIM Calculation
def calculate_ssim(img1, img2):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(img1, img2).item()

# LPIPS Calculation
def calculate_lpips(img1, img2):
    lpips_model = lpips.LPIPS(net='alex')
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    return lpips_model(img1, img2).item()
```

---

## **Future Enhancements**
- **Integrate Real-World Datasets**: Use tools like **COLMAP** to estimate camera poses for real-world images.
- **Optimize Rendering Speed**: Explore **Instant-NGP** for faster training and rendering.
- **Real-Time Applications**: Integrate NeRF with **robotics** for 3D mapping and navigation.

---

## 🧩 **Key Concepts to Understand**
| **Concept**              | **Description**                                             |
|--------------------------|-------------------------------------------------------------|
| **NeRF (Neural Radiance Fields)** | A neural network that predicts color and density in 3D space. |
| **Ray Marching**          | Simulating light rays traveling through a scene to sample points. |
| **Positional Encoding**   | Adding sine and cosine transformations to input coordinates for better spatial understanding. |
| **Camera Pose Matrix**    | Describes the position and orientation of the camera.       |
| **Volumetric Rendering**  | Rendering images by integrating color and density values along rays. |


