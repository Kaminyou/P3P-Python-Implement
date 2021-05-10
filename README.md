# P3P solver with RANSAC python implementation
This is the complete python implementation of p3p solver with RANSAC algorithm.
## Usage
`points2D`: a R<sup>4x2</sup> vector with numpy float type </br>
`points3D`: a R<sup>4x3</sup> vector with numpy float type
```python
from ransac import RANSACPnP
rvec, tvec, best_reproject_error = RANSACPnP(points2D, points3D, cameraMatrix, distCoeffs, times = args.iteration)
```