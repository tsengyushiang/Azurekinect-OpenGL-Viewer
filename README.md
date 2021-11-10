# Quick Start

- [Download Prebuilt libraries](https://drive.google.com/drive/folders/12qkwRdiAgQ-W6m_Nk8xAT9LLaW6u-mRJ?usp=sharing)

- unzip
  - `C:\git\pcl1.11.1`
  - `C:\git\imgui-1.82`
  - `C:\git\librealsense`
  - `C:\git\opencv`
  - `C:\git\opencv_contrib`
  - `C:\git\Azure Kinect SDK v1.4.1`

- add system ENV_PATH:
  - `C:\git\librealsense\build\Release`
  - `C:\git\PCL 1.11.1\bin`
  - `C:\git\PCL 1.11.1\3rdParty\VTK\bin`
  - `C:\git\PCL 1.11.1\3rdParty\OpenNI2\Redist`
  - `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin`

- open `app/app.sln` with visual studio 2019 swith to `Release x64`

# Setup libiraries from empty project

- [Premiere chromakey](./Premierechromakey.md)
- [opencv + realsense](./OpenCV-Realsesne.md)
- [imgui + opengl3](./Imgui-OpenGL3.md)
- [pcl](./pcl.md)
- [cuda v11.2](./cuda.md)
- [Realsense](./Realsense.md)
- [Azure Kinect](./azureKinect.md)

# Load External Data

+ CamData (example: data/1.json, data/2.json)
	+ include color image, depth map, intrinsic, resolution from single camera
	+ color map
		+ Int[] Colormap_raw 
			+ Range : 0~255
			+ Length : width * height * 3
	+ depth map
		+ *depth multiply scale = original depth (m)*
		+ Int[] depthmap_raw
			+ Length : width * height
		+ Float depthscale
	+ intrinsic
		+ Float fx, fy, ppx, ppy
	+ resolution
		+ int height, width

+ CamExtrinsics example: data/CameraExtrinsics.json
	+ include *ALL* cameras extrinsics & cam name
	+ float[16] camExtrinsic
	+ string camName