from matplotlib import pyplot as plt
import pandas as pd
import rasterio
import numpy as np
from shapely import Point
from sklearn.metrics import confusion_matrix, accuracy_score,  precision_score, recall_score, f1_score
from sklearn.neighbors import KDTree
import torch
import os
import re
from utility import load_model
from pyproj import CRS, Transformer
import seaborn as sns
from skimage.measure import label
from joblib import Parallel, delayed
import geopandas as gpd
import torch.nn.functional as F

# Script for running inference on subsequent linear regression over the AOI.
# For memory efficiency, it processes each image in four patches, and then reconstructs the full image from the patches.

NEW_MODEL = False
UNET = False
MODEL_NAME = "JACCARD" #Weights to use for inference, or any name for folder creation if NEW_MODEL is False
OUTPUT_DIR = "predictions"
MOSAIC_DIR = "mosaics"
THRESHOLD = 0

years = ['2017', '2018', '2019', '2020', '2021', '2022']
TILE_IDS = [84, 356, 336, 1051, 807, 170]
tile_names = ["Stavanger", "Oslo", "Trondheim", "Alta", "Tromsø", "Jostedalsbreen"]

# Read aoi from the shapefile
aoi_shapefile = "tiles_test.shp"
aoi_gdf = gpd.read_file(aoi_shapefile)

# Get all the geometries of the AOI
aoi_polygons = aoi_gdf['geometry'].tolist()

def save_trend_as_csv(model_name):
    if NEW_MODEL:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        model.eval()

        print(f"Model loaded")
        prediction_paths = save_prediction_tiffs_per_year_one_channel(model, device, years)
        
        torch.cuda.empty_cache()  # Clear GPU memory before processing each image
        print(f"GPU memory cleared", flush=True)
        del model # Free the model from memory
    else: # Load paths from the previous run
        prediction_paths = []
        for year in years:
            prediction_paths_current_year = []
            year_pred_dir = os.path.join(OUTPUT_DIR, str(year))
            for filename in os.listdir(year_pred_dir):
                if filename.endswith(".tif"):
                    prediction_paths_current_year.append(os.path.join(year_pred_dir, filename))
            prediction_paths.append(prediction_paths_current_year)
            
    print(f"Calculating trend image")
    writeTrendToCSV(prediction_paths)

# Function to save predictions as TIFF files for each year. Saves the argmax of the predictions as a single-channel TIFF.
def save_prediction_tiffs_per_year_one_channel(model, device, years):
    prediction_paths = []

    for year in years:
        prediction_paths_current_year = []
        tiff_dir = f"data/s2_summer_{year}"
        year_pred_dir = os.path.join(OUTPUT_DIR, str(year))
        os.makedirs(year_pred_dir, exist_ok=True)


        for filename in os.listdir(tiff_dir): 
            print(filename)  
            if not filename.endswith(".tif"):
                continue
            if not re.search(r"grid_cell_(84|170|336|356|807|1051)", filename):
                continue
            torch.cuda.empty_cache()  # Clear GPU memory before processing each image

            print(f"Loading image {filename} from {tiff_dir}", flush=True)
            path = os.path.join(tiff_dir, filename)

            # Open the image for inference
            with rasterio.open(path) as src:
                image = src.read()  # Read all bands
                transform = src.transform
                crs = src.crs

            image_tensor = torch.tensor(image, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)  # Move image to GPU
            
            # Split the large image into patches
            patches = split_image_into_patches(image_tensor)

            # Process each patch and store predictions
            all_predictions = []
            for patch in patches:
                print(f"Processing patch of shape {patch.shape}", flush=True)
                patch_width, patch_height = patch.shape[2], patch.shape[3]
                if UNET:
                    patch = pad_to_multiple(patch, multiple=32) # Pad the patch to be divisible by 32 for UNet
                patch = patch.to(device)
                with torch.no_grad():
                    torch.cuda.empty_cache() 
                    prediction = model(patch)
                    prediction = torch.argmax(prediction, dim=1)
                    
                prediction = prediction.squeeze(0).cpu().numpy()
                
                if UNET: # Remove padding if necessary
                    prediction = prediction[:patch_width, :patch_height] 

                all_predictions.append(prediction) 

            # Rebuild the full prediction from the patches
            full_prediction = reconstruct_image_from_patches_one_channel(all_predictions, image_tensor.shape)

            # Save the predictions as a TIFF file in the correct year directory
            print(f"Saving predictions for {filename} to {year_pred_dir}/{filename.split('.')[0]}_predictions.tif", flush=True)
            output_file = os.path.join(year_pred_dir, f"{filename.split('.')[0]}_predictions.tif")
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=image.shape[1],
                width=image.shape[2],
                count= 1, # Single-channel prediction
                dtype=np.float32,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(full_prediction, 1)
            
            prediction_paths_current_year.append(output_file)
            print(f"Saved predictions to {output_file}", flush=True)
            
        prediction_paths.append(prediction_paths_current_year)

    return prediction_paths

# Trial function used for linear regression on the probability scores.
# Not used in final implementation, but kept for reference.
def save_prediction_tiffs_per_year_probs(model, device, years):
    prediction_paths = []

    for year in years:
        prediction_paths_current_year = []
        tiff_dir = f"data/s2_summer_{year}"
        year_pred_dir = os.path.join(OUTPUT_DIR, str(year))
        os.makedirs(year_pred_dir, exist_ok=True)

        for filename in os.listdir(tiff_dir): 
            print(filename)  
            if not filename.endswith(".tif"):
                continue
            if not re.search(r"grid_cell_(84|170|336|356|807|1051)", filename):
                continue
            torch.cuda.empty_cache()

            print(f"Loading image {filename} from {tiff_dir}", flush=True)
            path = os.path.join(tiff_dir, filename)

            with rasterio.open(path) as src:
                image = src.read()
                transform = src.transform
                crs = src.crs

            image_tensor = torch.tensor(image, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)

            # Split the large image into patches
            patches = split_image_into_patches(image_tensor)

            # Process each patch and store predictions
            all_probabilities = []
            for patch in patches:
                patch = patch.to(device) 
                with torch.no_grad():
                    torch.cuda.empty_cache() 
                    prediction = model(patch)

                all_probabilities.append(prediction.squeeze(0).cpu().numpy())  # Save probabilities for all classes

            # Rebuild the full prediction from the patches
            full_prediction = reconstruct_image_from_patches(all_probabilities, image_tensor.shape)
            
            channels = full_prediction.shape[0]

            # Save the predictions as a TIFF file in the correct year directory
            print(f"Saving predictions for {filename} to {year_pred_dir}/{filename.split('.')[0]}_predictions.tif", flush=True)
            output_file = os.path.join(year_pred_dir, f"{filename.split('.')[0]}_predictions.tif")
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=image.shape[1],
                width=image.shape[2],
                count= 2, # Single-channel prediction
                dtype=np.float32,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(full_prediction[6], 1)

            prediction_paths_current_year.append(output_file)
            print(f"Saved predictions to {output_file}", flush=True)
            
        prediction_paths.append(prediction_paths_current_year)

    return prediction_paths

def pad_to_multiple(tensor, multiple=32):
    b, c, h, w = tensor.size()
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

def split_image_into_patches(image_tensor):
    patches = []
    print(f"Splitting image tensor of shape {image_tensor.shape} into patches", flush=True)
    
    # Get the shape of the image
    batch_size, channels, height, width = image_tensor.shape

    # Calculate the patch size
    patch_height = height // 2
    patch_width = width // 2

    # Loop through the image and split into patches
    for i in range(0, height - 1, patch_height):
        for j in range(0, width - 1, patch_width):
            end_i = min(i + patch_height, height)
            end_j = min(j + patch_width, width)  
            
            # If image dimensions are not perfectly divisible by patch size, adjust the end indices
            if height - end_i == 1: 
                end_i = height
            if width - end_j == 1:   
                end_j = width 
            
            patch = image_tensor[:, :, i:end_i, j:end_j]  # Slice the image to get the patch      
            
            print(f"Patch created with shape: {patch.shape}", flush=True)
            patches.append(patch)

    return patches


# Function to reconstruct the full image from patches.
def reconstruct_image_from_patches_one_channel(patches, image_shape):
    print(f"Reconstructing image from {len(patches)} patches", flush=True)
    print(f"Image shape: {image_shape}", flush=True)
    
    _, _, full_height, full_width = image_shape
    full_image = np.zeros((full_height, full_width), dtype=np.float32) 
    patch_idx = 0
    i = 0
    j = 0

    while i < full_height:
        while j < full_width and patch_idx < len(patches):
            patch = patches[patch_idx]
            print(f"Patch shape: {patch.shape}", flush=True)
            patch_height, patch_width = patch.shape
            
            # Insert the patch into the full image
            full_image[i:i + patch_height, j:j + patch_width] = patch 

            j += patch_width 
            patch_idx += 1
    
   
        i += patch_height
        j = 0 

    return full_image

# Function to reconstruct the full image from patches, in the case of using multi-channel images.
# Not used in the final implementation, but kept for reference.
def reconstruct_image_from_patches(patches, image_shape):
    print(f"Reconstructing image from {len(patches)} patches", flush=True)
    print(f"Image shape: {image_shape}", flush=True)
    
    full_depth, full_height, full_width = image_shape
    full_image = np.zeros((full_depth, full_height, full_width), dtype=np.float32)  
    patch_idx = 0
    i = 0
    j = 0

    while i < full_height:
        while j < full_width and patch_idx < len(patches):
            patch = patches[patch_idx]
            print(f"Patch shape: {patch.shape}", flush=True)
            _, patch_height, patch_width = patch.shape
            
            full_image[:, i:i + patch_height, j:j + patch_width] = patch 

            j += patch_width
            patch_idx += 1
    
        i += patch_height
        j = 0 

    return full_image

def getLinearCoeff(images):
    print(f"Stacking {len(images)} images", flush=True)
    print(f"Image shape: {images[0].shape}", flush=True)

    image_stack = np.stack(images, axis=0)

    # Get the number of time steps (images) and dimensions of each image
    print(f"Image stack shape: {image_stack.shape}", flush=True)
    _, rows, cols = image_stack.shape
    time_steps = len(images)
    years_since_start = np.arange(time_steps)

    print(f"Performing linear regression for {rows * cols} pixels", flush=True)

    trend = np.zeros((rows, cols), dtype=int)

    # Run the pixel processing in parallel
    trend = Parallel(n_jobs=-1)(delayed(process_pixel)(i, j, image_stack, years_since_start) for i in range(rows) for j in range(cols))

    # Reshape the result back into the correct shape (rows, cols)
    trend = np.array(trend).reshape((rows, cols))

    return trend


# Change classes to reduce complexity of image. Filters out all but the built class.
def changeTypology(img):
    built = np.where(img == 6, 1, 0)  # Built
    built = np.squeeze(built, axis=0) 
    
    return built
    

def process_pixel(i, j, image_stack, years_since_start):
    pixel_values = image_stack[:, i, j]

    valid_indices = ~np.isnan(pixel_values)  # Indices where pixel values are not NaN

    if np.sum(valid_indices) < 4: # Set slope to 0 if fewer than 4 valid values
        return 0
    
    if np.any(valid_indices):  # Only run regression if there are valid values
        slope, intercept = np.polyfit(years_since_start[valid_indices], pixel_values[valid_indices], 1)
    else:
        print(f"No valid values for pixel ({i}, {j})", flush=True)
        return 0
    
    slope = int(round(slope * 100)) # Scaling slope for better readability

    return slope


def getTrendImg(area_paths):
    crs = None
    transform = None
    image_stack = []
    for path in area_paths:
        with rasterio.open(path) as src:
            image = src.read()
            image_stack.append(image)
            
            if crs is None:
                crs = src.crs
            if transform is None:
                transform = src.transform
                
    print(f"Changing typology of {len(image_stack)} images", flush=True)
    for idx, image in enumerate(image_stack):
        image_stack[idx] = changeTypology(image)
        
    # Perform linear regression on the images
    print(f"Calculating trend for {len(image_stack)} images", flush=True)
    trend = getLinearCoeff(image_stack)
    
    
    return trend, crs, transform

def writeTrendToCSV(prediction_paths):
    output_file = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_trend.csv")
    
    tile_ids = []
    for filename in prediction_paths[0]:
        match = re.search(r"grid_cell_(\d+)_predictions\.tif$", filename)
        if match:
            tile_ids.append(int(match.group(1)))
    tile_ids = sorted(set(tile_ids))
    
    with open(output_file, 'w') as f:
        f.write("lat,lon,trend\n")
        for tile_id in tile_ids:
            tile_prediction_paths = []
            for year in years:
                year_pred_dir = os.path.join(OUTPUT_DIR, str(year))
                for filename in os.listdir(year_pred_dir):
                    match = re.match(rf"s2_summer_{year}_grid_cell_{tile_id}_predictions\.tif$", filename)
                    if match:
                        tile_prediction_paths.append(os.path.join(year_pred_dir, filename))
                    else:
                        print(f"File {filename} does not match the expected pattern for tile {tile_id}", flush=True)
            print(f"Found {len(tile_prediction_paths)} predictions for tile {tile_id}", flush=True)
            
            # Calculate the trend for the current tile
            trend, crs, transform = getTrendImg(tile_prediction_paths)
            
            wgs84_proj = CRS.from_epsg(4326)
            image_proj = CRS.from_string(str(crs))
            transformer = Transformer.from_crs(image_proj, wgs84_proj, always_xy=True)
            
            print(f"Maximum trend value for tile {tile_id}: {np.max(trend)}", flush=True)
            
            for i in range(trend.shape[0]):
                for j in range(trend.shape[1]):
                    x, y = rasterio.transform.xy(transform, i, j)
                    lon, lat = transformer.transform(x, y)
                    
                    f.write(f"{lat},{lon},{trend[i][j]}\n") # Write the trend value for each pixel
            
            print(f"Processed tile {tile_id} with trend values", flush=True)

def compare_to_test_data():
    print("Loading test data", flush=True)
    validation_path = "data/test_pixel_level.csv"
    validation_df = pd.read_csv(validation_path)
    
    print("Loading prediction data", flush=True)
    prediction_path = f"predictions/{MODEL_NAME}_trend.csv"
    prediction_df = pd.read_csv(prediction_path)
    
    # Extract the latitude and longitude from the prediction data
    prediction_coords = prediction_df[['lat', 'lon']].values
    
    print("Building KDTree", flush=True)
    tree = KDTree(prediction_coords)
    
    merged_data = []

    # Iterate over each pixel in the validation data to match it with the closest prediction
    for _, test_row in validation_df.iterrows():
        test_lat = test_row['latitude']
        test_lon = test_row['longitude']
        
        test_point = Point(test_lon, test_lat) 
        
        if not any(aoi_polygon.contains(test_point) for aoi_polygon in aoi_polygons):
            continue  # Skip this test point if it's not within any AOI polygon
        else: # Find the tile ID
            for _, polygon in aoi_gdf.iterrows():
                if polygon['geometry'].contains(test_point):
                    tile_id = polygon['tile_id']
                    break
        
        print(f"Processing test point: {test_lat}, {test_lon}")
       
        test_point = np.array([[test_lat, test_lon]])
        
        # Query the KDTree for the closest prediction (returning the index of the closest point)
        dist, idx = tree.query(test_point, k=1)
        
        # Get the closest trend value and its corresponding coordinates
        closest_trend = prediction_df.iloc[idx[0][0]]['trend']
        closest_lat = prediction_df.iloc[idx[0][0]]['lat']
        closest_lon = prediction_df.iloc[idx[0][0]]['lon']
        
        merged_data.append([test_row['sampleID'], test_lat, test_lon, test_row['label'], closest_trend, closest_lat, closest_lon, tile_id])

    merged_df = pd.DataFrame(merged_data, columns=["sampleID", "latitude", "longitude", "label", "trend", "predicted_lat", "predicted_lon", "tile_id"])
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv("predictions/merged_validation.csv", index=False)
    print("Merged DataFrame with closest predictions and coordinates saved to predictions/merged_validation.csv")
    
    return merged_df

def make_confusion_matrix_for_tile(merged_df, tile_id):
    # Filter the merged data to include only the current tile
    tile_data = merged_df[merged_df['tile_id'] == tile_id]

    if tile_data.empty:
        return None  # If no data for this tile, return None

    cm = confusion_matrix(tile_data['label'], tile_data['trend'])
    
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    
    true_label = tile_data['label']
    predicted_label = tile_data['trend']
    
    oa = accuracy_score(true_label, predicted_label)
    precision = precision_score(true_label, predicted_label, pos_label='loss')
    recall = recall_score(true_label, predicted_label, pos_label='loss')
    f1 = f1_score(true_label, predicted_label, pos_label='loss')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
   
    metrics = {
        "model_name": MODEL_NAME,
        "tile_id": tile_id,
        "overall_accuracy": oa,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results/evaluation_metrics_tiles.csv", index=False, mode='a', header=not pd.io.common.file_exists(f"results/evaluation_metrics_tiles.csv"))
    print(f"Evaluation metrics for tile {tile_id} saved to results/evaluation_metrics_tile_{tile_id}.csv", flush=True)
    
    return cm
    

def make_total_confusion_matrix(merged_df):
    """  
    Create a confusion matrix using sklearn to compare the predicted trends with the actual labels.
    """
    
    # Turn continuous values into discrete classes
    merged_df['trend'] = merged_df['trend'].apply(lambda x: 'stable' if x <= THRESHOLD else 'loss')
    
    cm = confusion_matrix(merged_df['label'], merged_df['trend'])
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 28})
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['loss', 'stable'], yticklabels=['loss', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.savefig(f"results/{MODEL_NAME}_confusion_matrix.png", dpi=300)
    print(f"Confusion matrix saved to results/{MODEL_NAME}_confusion_matrix.png", flush=True)
    
    return cm.ravel() # Flatten the confusion matrix for evaluation metrics

def plot_confusion_matrices_for_tiles(merged_df):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    axes = axes.flatten()

    # Loop through each tile and plot its confusion matrix
    for i, tile_id in enumerate(TILE_IDS):
        cm = make_confusion_matrix_for_tile(merged_df, tile_id)
        
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['loss', 'stable'], yticklabels=['loss', 'stable'], ax=axes[i])
            axes[i].set_title(f'{tile_names[i]}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/{MODEL_NAME}_confusion_matrices_all_tiles.png", dpi=300)
    print(f"Confusion matrices for all tiles saved to results/{MODEL_NAME}_confusion_matrices_all_tiles.png", flush=True)

def calculate_evaluation_metrics(df, cm): 
    true_label = df['label']
    predicted_label = df['trend']

    accuracy = accuracy_score(true_label, predicted_label)
    precision = precision_score(true_label, predicted_label, pos_label='loss')
    recall = recall_score(true_label, predicted_label, pos_label='loss')
    f1 = f1_score(true_label, predicted_label, pos_label='loss')
    specificity = cm[0] / (cm[0] + cm[1])

    results_df = pd.DataFrame({
        "Model Name": [MODEL_NAME],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Specificity": [specificity]
    })
    
    results_df.to_csv("results/evaluation_metrics_change_detection.csv", 
                      mode='a', header=not pd.io.common.file_exists("results/evaluation_metrics_change_detection.csv"), index=False)
    print("Evaluation metrics saved to results/evaluation_metrics_change_detection.csv")
    
def validate_built_probability():

    merged_df = compare_to_test_data()
    
    cm = make_total_confusion_matrix(merged_df)
    plot_confusion_matrices_for_tiles(merged_df)
    calculate_evaluation_metrics(merged_df, cm)
    
save_trend_as_csv(MODEL_NAME)
validate_built_probability()

