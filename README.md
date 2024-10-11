# Medical Image Segmentation API

## Overview
This project implements a RESTful API for medical image segmentation, using the **Segment Anything Model (SAM)** developed by Meta. The API is built with **FastAPI** and is designed to integrate with the **OHIF** (Open Health Imaging Foundation) medical imaging viewer. It allows healthcare professionals to upload medical images, perform image segmentation, and retrieve results in DICOM or PNG format.

The API is designed to run in a secure, internal environment within healthcare institutions, ensuring compliance with medical data privacy regulations.

## Key Features
- **FastAPI-based RESTful API** for handling segmentation requests.
- **SAM (Segment Anything Model)** for automatic image segmentation.
- Support for **DICOM** and **PNG** formats.
- **Internal deployment** to ensure data privacy within the institution's network.
- **Interoperability** with the OHIF viewer, providing a seamless integration for medical professionals.

## Technology Stack
- **FastAPI**: Framework used for creating the API.
- **SAM (Segment Anything Model)**: Model used for medical image segmentation.
- **Python**: Programming language used for backend development.
- **Docker**: For containerization and easy deployment.
  
## API Endpoints

### 1. `/segment_input_image/` (POST)
- **Description**: Receives an image and segmentation parameters, performs the segmentation, and returns a timestamp for tracking the request.
- **Request**:
  - **Image file**: Medical image in supported formats (e.g., PNG, JPEG).
  - **JSON**: Parameters such as positive/negative points and rectangles for guiding the segmentation process.
- **Response**:
  - JSON containing a `timestamp` to track the segmentation request.

### 2. `/get_segmented_image/{timestamp}` (GET)
- **Description**: Retrieves the segmented image in PNG format using the `timestamp`.
- **Parameters**:
  - `timestamp`: Identifier for the segmentation process.
- **Response**:
  - PNG file of the segmented image.

### 3. `/get_dicom_image/{timestamp}` (GET)
- **Description**: Retrieves the segmented image in DICOM format.
- **Parameters**:
  - `timestamp`: Identifier for the segmentation process.
- **Response**:
  - DICOM file containing the segmented image.

### 4. `/health_check/` (GET)
- **Description**: API health check, confirming the service is running.
- **Response**:
  - JSON with the API's status.

## Architecture
The API follows a **client-server architecture**, where the client (OHIF viewer) sends requests to the API to perform segmentation. The API processes the images using the SAM model and returns the segmented results.

- **OHIF Viewer**: Frontend viewer for interacting with the images.
- **FastAPI**: Backend framework for handling requests.
- **SAM Model**: Performs the image segmentation.
- **Docker**: Ensures easy deployment across various environments.

## Installation

### Prerequisites
- **Docker**: Ensure Docker is installed on your system.
- **Python 3.8+**: Required to run the API.
- **Pip**: Python package manager.

### Steps to Install and Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sergiotm13/SAM-API.git
   cd SAM-API
    ```
   
2. **Create a Virtual Environment (optional but recommended):**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**

   ```bash
   uvicorn main
   ```

## Data Models
The API uses several data models to validate and manage incoming data for the segmentation process:

1. **SAMInput**:
   - This model handles the primary inputs required for a segmentation request.
   - **Attributes**:
     - `model`: Specifies the model to be used for segmentation (e.g., SAM).
     - `rectangles`: A list of `RectInput` objects defining areas for segmentation.
     - `positive_points`: A list of `PointInput` objects marking areas of interest.
     - `negative_points`: A list of `PointInput` objects marking areas to exclude from segmentation.

2. **RectInput**:
   - Defines rectangular areas within the image to guide the segmentation model.
   - **Attributes**:
     - `startX`: X-coordinate of the rectangle's top-left corner.
     - `startY`: Y-coordinate of the rectangle's top-left corner.
     - `width`: Width of the rectangle.
     - `height`: Height of the rectangle.

3. **PointInput**:
   - Specifies a single point within the image.
   - **Attributes**:
     - `x`: X-coordinate of the point.
     - `y`: Y-coordinate of the point.

Each of these models is validated upon request submission, ensuring the API processes well-structured data. 

## Security and Privacy
This API is intended for use in secure healthcare environments, ensuring compliance with data protection regulations:
- **CORS Middleware**: Configured to restrict API access to trusted origins. By default, it allows access from `http://localhost` and `http://localhost:3000` to facilitate local testing and development.
- **Data Handling**: Data is stored with timestamp identifiers for ease of tracking, retrieval, and ensuring the traceability of each segmentation process.

## Future Enhancements
Future updates to the API may include:
- **Support for additional segmentation models**: Expand model options to enhance flexibility.
- **Enhanced parameter configurations**: Allow users to define more specific options for tailored segmentation.
- **Monitoring and logging**: Implement comprehensive logging for auditing purposes, which is crucial in healthcare applications.

## Contributions
Contributions are welcome to improve and expand the project. To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Make your changes and submit a pull request with a description of the modifications.

## License
This project is distributed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.

