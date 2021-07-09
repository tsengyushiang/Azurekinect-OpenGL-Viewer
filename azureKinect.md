# win10 vs2019 Azure Kinect

- [Download window SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)

    - The installer will put all the needed headers, binaries, and tools in the location you choose (by default this is `C:\Program Files\Azure Kinect SDK version\sdk`

    - In my case I copy all files to `C:\git\Azure Kinect SDK version\sdk`

- Open vs2019 

    - Include `C:\git\Azure Kinect SDK v1.4.1\sdk\include`
    
    - Add lib `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib\k4a.lib`

- Add DLL to System Path

    - `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin`

- Tutorials :

    - [vs2019 Setting](https://blog.csdn.net/hanshuning/article/details/112555140)

    - [offical startup](https://pterneas.com/2020/03/19/azure-kinect/)