# app.py

import os
import subprocess
from time import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
    Orientation,
    Spacing,
    NormalizeIntensity,
    Activations,
    AsDiscrete
)
from BrainTumorDataset import BrainTumorDataset
import torch
# DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")
app = Flask(__name__)

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),"segmentations")
ROI_MARGIN = '[20,8,20]'
CONTRAST = 't1'
SEG_CA_MODE = '1/2/3'
# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_metric_model.pth")
app.config['MODEL_PATH'] = MODEL_PATH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit the maximum file size to 16MB

app.config['OUTPUT_FOLDER'] = OUTPUT_ROOT
# Allowed extension for file uploads
ALLOWED_EXTENSION = 'nii.gz'


def allowed_file(filename):
    """Check if the uploaded file has the allowed extension."""
    return filename.endswith('.' + ALLOWED_EXTENSION)


def resample_image_to_square_grid(image):
    """Resample the image to a square grid size with isotropic spacing."""
    # Get the original size and spacing
    original_size = np.array(image.GetSize())
    original_spacing = np.array(image.GetSpacing())

    # Determine the target size (max dimension) and calculate new spacing
    target_size = np.max(original_size)
    new_spacing = original_spacing * (original_size / target_size)

    # Create the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([int(target_size)] * 3)  # Target size is a cube
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)

    # Resample the image
    resampled_image = resampler.Execute(image)
    return resampled_image


def segment_hippocampus(image_path):
    absolute_path = os.path.abspath(os.path.expanduser(os.path.join(UPLOAD_FOLDER, 'hipp')))
    print(absolute_path)
    # Define the command and arguments
    command = "hsf"
    args = [
        f'files.path={absolute_path}',
        'files.pattern="*.nii.gz"',
        f'roiloc.margin={ROI_MARGIN}',
        f'roiloc.contrast="{CONTRAST}"',
        f'segmentation.ca_mode="{SEG_CA_MODE}"'
    ]

    # Combine the command and arguments into a single command string
    cmd = f"{command} {' '.join(args)}"
    print("Running command: ", cmd)
    start_time = time()
    # Run the command
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time()
    time_elapsed = (end_time - start_time) / 60
    print("Time elapsed: ", time_elapsed, " mins")
    # Print the output and error (if any)
    print("Output:")
    print(process.stdout)
    if process.stderr:
        print("Error:")
        print(process.stderr)
        return "ERROR: " + str(process.stderr)

    sub_name = image_path.split(os.sep)[-1].split('.')[0]
    left_seg_file = str(os.path.join(os.path.abspath(UPLOAD_FOLDER), 'hipp', sub_name + '_left_hippocampus_seg.nii.gz'))
    right_seg_file = str(os.path.join(os.path.abspath(UPLOAD_FOLDER), 'hipp', sub_name + '_right_hippocampus_seg.nii.gz'))
    try:
        left_seg = sitk.ReadImage(left_seg_file)
        right_seg = sitk.ReadImage(right_seg_file)
        combined_image = sitk.Or(left_seg, right_seg)

        segmented_file_path = str(os.path.join(app.config['OUTPUT_FOLDER'], sub_name + '_seg.nii.gz'))
        segmented_file_name = sub_name + '_seg.nii.gz'
        print("Writing segmented image in: ", segmented_file_path)
        sitk.WriteImage(combined_image, segmented_file_path)
        return segmented_file_name
    except RuntimeError as err:
        print(err)
        return "ERROR "+str(err)

# define inference method
def tumor_inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    return _compute(input)
def segment_tumor(image_path: str):
    absolute_path = os.path.abspath(os.path.expanduser(os.path.join(UPLOAD_FOLDER, 'hipp')))
    print(absolute_path)
    sub_name = image_path.split(os.sep)[-1].split('.')[0]
    test_transform = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            EnsureType(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ]
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    test_ds = BrainTumorDataset([image_path], transform=test_transform)
    test_image = test_ds[0].unsqueeze(0).to(DEVICE)
    model = AttentionUnet(spatial_dims=3,
                          in_channels=1,
                          out_channels=1,
                          channels=(16, 32, 64, 128, 256),
                          strides=(2, 2, 2, 2),
                          dropout=0.2,
                          ).to(DEVICE)
    # model.load_state_dict(torch.load(app.config['MODEL_PATH'], weights_only=True, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(app.config['MODEL_PATH'], weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    print("Model Loaded")
    try:
        with torch.no_grad():
            print("Segmenting")
            output = tumor_inference(test_image, model)
            output = post_trans(output[0])
            image_array = test_image.detach().cpu().numpy()
            output_array = output.detach().cpu().numpy()
            image_array = np.squeeze(np.squeeze(image_array))
            output_array = np.squeeze(np.squeeze(output_array))
            # print(image_array.shape)
            # print(output_array.shape)
            img_sitk = sitk.GetImageFromArray(np.moveaxis(image_array, [0,1,2], [1,2,0]))
            output_sitk = sitk.GetImageFromArray(np.moveaxis(output_array, [0,1,2], [1,2,0]))
            output_path = str(os.path.join(app.config['OUTPUT_FOLDER'], sub_name + '_seg.nii.gz'))
            # input_origin = img_sitk.GetOrigin()
            # output_sitk.SetOrigin(input_origin)
            # output_sitk.SetSpacing((1.0, 1.0, 1.0))
            segmented_file_name = sub_name + '_seg.nii.gz'
            print("Writing segmented image in: ", output_path)
            sitk.WriteImage(output_sitk, output_path)
            sitk.WriteImage(img_sitk, image_path)
            return segmented_file_name
    except Exception as err:
        print(err)
        return "ERROR " + str(err)
@app.route('/')
def home():
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'hipp')):
        print("Upload folder for hippocampus segmentations do not exist")
        print("Creating ", os.path.join(UPLOAD_FOLDER, 'hipp'))
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'hipp'))
    if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'tumor')):
        print("Upload folder for brain tumor segmentations do not exist")
        print("Creating ", os.path.join(UPLOAD_FOLDER, 'tumor'))
        os.makedirs(os.path.join(UPLOAD_FOLDER, 'tumor'))
    return render_template('index.html')
    # return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    # Assumes the segmentations are stored in a folder named 'segmentations' at the same level as templates
    # directory = os.path.join(app.config['OUTPUT_FOLDER'])
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/download_sample/<filename>')
def download_sample_file(filename):
    # Assumes the segmentations are stored in a folder named 'segmentations' at the same level as templates
    # directory = os.path.join(app.config['OUTPUT_FOLDER'])
    return send_from_directory("sample_data", filename, as_attachment=True)

@app.route('/error')
def error():
    error_msg = request.args.get('error_msg')
    return render_template('error.html', error_msg=error_msg)

@app.route('/hipp', methods=['POST'])
def upload_hipp():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Securely save the file
            filename = file.filename

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hipp', filename)
            file.save(file_path)
            segmented_file_name = segment_hippocampus(file_path)
            if not segmented_file_name.startswith("ERROR"):
                # Redirect to the visualization page with the filename
                return redirect(url_for('visualize', filename=segmented_file_name, task='hipp'))
            else:
                return redirect(url_for('error', error_msg=segmented_file_name))
        else:
            return "Only '.nii.gz' files are allowed."
    return render_template('index.html')

@app.route('/tumor', methods=['POST'])
def upload_tumor():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Securely save the file
            filename = file.filename

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tumor', filename)
            file.save(file_path)
            # return render_template('coming_soon.html')
            segmented_file_name = segment_tumor(file_path)
            if not segmented_file_name.startswith("ERROR"):
                # Redirect to the visualization page with the filename
                return redirect(url_for('visualize', filename=segmented_file_name, task='tumor'))
            else:
                return redirect(url_for('error', error_msg=segmented_file_name))
        else:
            return "Only '.nii.gz' files are allowed."
    return render_template('index.html')

@app.route('/visualize/<filename>')
def visualize(filename: str):
    """Visualize the NIfTI file using 2D and 3D rendering."""
    task = request.args.get('task', default='hipp', type=str)
    # Load the NIfTI file using SimpleITK
    seg_file_path = str(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    input_file_name = filename.replace('_seg', '')
    input_file_path = str(os.path.join(app.config['UPLOAD_FOLDER'], task, input_file_name))
    print("Loading Images")

    seg_img = sitk.ReadImage(seg_file_path)
    input_img = sitk.ReadImage(input_file_path)
    print("Input Image")
    print(input_img.GetSize())
    print(input_img.GetOrigin())
    print(input_img.GetSpacing())
    print(input_img.GetDimension())
    print("Segmented Image")
    print(seg_img.GetSize())
    print(seg_img.GetOrigin())
    print(seg_img.GetSpacing())
    print(seg_img.GetDimension())
    # Resample the image to a square grid
    print("Resampling Images")
    resampled_seg_img = resample_image_to_square_grid(seg_img)
    resampled_input_img = resample_image_to_square_grid(input_img)

    seg_data = sitk.GetArrayFromImage(resampled_seg_img)
    input_data = sitk.GetArrayFromImage(resampled_input_img)

    # Prepare volume data for 3D rendering
    print("Preparing volumetric data")
    volume_fig = go.Figure(data=go.Volume(
        x=np.arange(0, seg_data.shape[2]),
        y=np.arange(0, seg_data.shape[1]),
        z=np.arange(0, seg_data.shape[0]),
        value=seg_data.flatten(),
        opacity=0.1,  # Adjust opacity for better visualization
        surface_count=15,  # Number of isosurfaces to display
        colorscale='Viridis',  # Choose a colorscale
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    # Convert the Plotly figure to JSON for rendering in the HTML template
    volume_json = volume_fig.to_json()

    # Pass the data shape to the template for slider limits
    data_shape = seg_data.shape

    # Convert data to a list for easier use in JavaScript
    seg_data_list = seg_data.tolist()
    input_data_list = input_data.tolist()

    # Return the rendered template
    return render_template(
        'visualize.html',
        data_shape=data_shape,
        seg_data_list=seg_data_list,
        input_data_list=input_data_list,
        volume_json=volume_json,
        filename=filename
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=85)
