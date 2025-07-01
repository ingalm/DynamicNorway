import pandas as pd
import numpy as np
import rasterio
import os
from shapely import Point
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,  precision_score, recall_score, f1_score
from sklearn.neighbors import KDTree
import seaborn as sns
import geopandas as gpd

# Script used to evaluate the results from the global DW model from Google.
# Uses predictions exported from GEE.

tiles_to_include = [84, 170, 336, 356, 807, 1051]

path_to_tiffs = "predictions/DWnedbygging"
aoi = "tiles_test.shp"  # Path to the shapefile containing the AOI

# Filter tiff paths to tiles_to_include
tiff_paths = [os.path.join(path_to_tiffs, file) for file in os.listdir(path_to_tiffs) if file.endswith(".tif")]
tiff_paths = [tiff_path for tiff_path in tiff_paths if any(str(tile) in tiff_path for tile in tiles_to_include)]

# Limit aoi to  the tiles_to_include
aoi_gdf = gpd.read_file(aoi)
aoi_gdf = aoi_gdf[aoi_gdf['tile_id'].isin(tiles_to_include)]

print(f"Found {len(tiff_paths)} TIFF files in {path_to_tiffs}", flush=True)
print(f"Tile_ids in AOI after filtering: {aoi_gdf['tile_id'].unique()}", flush=True)

# Function to write pixels from DW TIFFs to a CSV file
def write_pixels_from_nedbygging_tiffs(tiff_paths):
    output_csv = "predictions/DW_nedbygging_pixels.csv"
    
    for tiff_path in tiff_paths:
        with rasterio.open(tiff_path) as src:
            print(f"Processing {tiff_path}")
            data = src.read(1)
            transform = src.transform
            
            print(f"Data shape: {data.shape}")
            
            print(f"Data shape: {data.shape}")
            
            # Initialize chunks size
            chunk_size = 1000  # Adjust this depending on your system's capabilities
            
            # Loop through the data in chunks
            for start_row in range(0, data.shape[0], chunk_size):
                end_row = min(start_row + chunk_size, data.shape[0])
                
                print(f"Processing rows from {start_row} to {end_row}", flush=True)
                
                chunk_data = data[start_row:end_row, :]
                
                # Get the coordinates of the pixels for this chunk
                rows, cols = np.indices(chunk_data.shape)
                
                # Convert row/col indices to geographic coordinates (lon, lat)
                lon, lat = rasterio.transform.xy(transform, rows + start_row, cols)
                
                lon = np.array(lon).flatten()
                lat = np.array(lat).flatten()
                values = chunk_data.flatten()
                
                pixels_df = pd.DataFrame({
                    'lon': lon,
                    'lat': lat,
                    'value': values
                })
                
                 # Write to CSV incrementally
                pixels_df.to_csv(output_csv, index=False, header=True, mode='w')
            
                print(f"Wrote {len(pixels_df)} pixels from rows {start_row} to {end_row}.", flush=True)
    
    print("All pixels saved to predictions/DW_nedbygging_pixels.csv", flush=True)
    

def compare_to_test_data():
    aoi_polygons = aoi_gdf['geometry'].tolist()

    print("Loading validation and prediction data...", flush=True)
    validation_path = "data/test_pixel_level.csv"
    validation_df = pd.read_csv(validation_path)
    
    prediction_path = "predictions/DW_nedbygging_pixels.csv"
    prediction_df = pd.read_csv(prediction_path)

    prediction_coords = prediction_df[['lat', 'lon']].values 

    print("Building KDTree for prediction coordinates...", flush=True)
    tree = KDTree(prediction_coords)
    
    merged_data = []

    # Iterate over each pixel in the validation data and find the closest prediction
    for _, test_row in validation_df.iterrows():
        test_lat = test_row['latitude']
        test_lon = test_row['longitude']
        
        test_point = Point(test_lon, test_lat)
        
        tile_id = None
        
        # Check if the test point is within any of the AOI polygons
        if not any(aoi_polygon.contains(test_point) for aoi_polygon in aoi_polygons):
            continue  # Skip this test point if it's not within any AOI polygon
        else: # Find the tile ID
            for _, polygon in aoi_gdf.iterrows():
                if polygon['geometry'].contains(test_point):
                    tile_id = polygon['tile_id']
                    break
       
        test_point = np.array([[test_lat, test_lon]])
        
        # Query the KDTree for the closest prediction (returning the index of the closest point)
        dist, idx = tree.query(test_point, k=1) 
        
        # Get the closest trend value and its corresponding coordinates
        closest_trend = prediction_df.iloc[idx[0][0]]['value']
        closest_lat = prediction_df.iloc[idx[0][0]]['lat']
        closest_lon = prediction_df.iloc[idx[0][0]]['lon']
        
        print(f"Test pixel: {test_row['sampleID']} - Closest prediction: {closest_trend} at ({closest_lat}, {closest_lon})", flush=True)
        
        merged_data.append([test_row['sampleID'], test_lat, test_lon, test_row['label'], closest_trend, closest_lat, closest_lon, tile_id])
    
    merged_df = pd.DataFrame(merged_data, columns=["sampleID", "latitude", "longitude", "label", "trend", "predicted_lat", "predicted_lon", "tile_id"])
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv("predictions/merged_validation.csv", index=False)
    print("Merged DataFrame with closest predictions and coordinates saved to predictions/merged_validation.csv", flush=True)
    
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
        "model_name": "dw_real",
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
    # Turn continous values into discrete classes
    merged_df['trend'] = merged_df['trend'].apply(lambda x: 'stable' if x <= 0 else 'loss')
    
    cm = confusion_matrix(merged_df['label'], merged_df['trend'])
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 24})
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['loss', 'stable'], yticklabels=['loss', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.savefig("results/confusion_matrix.png", dpi=300)
    print("Confusion matrix saved to results/confusion_matrix.png", flush=True)
    
    return cm.ravel() # Flatten the confusion matrix for evaluation metrics

def plot_confusion_matrices_for_tiles(merged_df):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    axes = axes.flatten()

    # Loop through each tile and plot its confusion matrix
    for i, tile_id in enumerate(tiles_to_include):
        cm = make_confusion_matrix_for_tile(merged_df, tile_id)
        
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['loss', 'stable'], yticklabels=['loss', 'stable'], ax=axes[i])
            axes[i].set_title(f'Tile {tile_id}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("results/confusion_matrices_all_tiles.png", dpi=300)
    print("Confusion matrices for all tiles saved to results/confusion_matrices_all_tiles.png", flush=True)


def calculate_evaluation_metrics(df, cm):
    true_label = df['label']
    predicted_label = df['trend']

    accuracy = accuracy_score(true_label, predicted_label)
    precision = precision_score(true_label, predicted_label, pos_label='loss')
    recall = recall_score(true_label, predicted_label, pos_label='loss')
    f1 = f1_score(true_label, predicted_label, pos_label='loss')
    specificity = cm[0] / (cm[0] + cm[1])

    results_df = pd.DataFrame({
        "Model Name": ["DWnedbygging"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Specificity": [specificity]
    })
   
    results_df.to_csv("results/evaluation_metrics_change_detection.csv", index=False, mode='a', header=not pd.io.common.file_exists("results/evaluation_metrics_change_detection.csv"))
    print("Evaluation metrics saved to results/evaluation_metrics_change_detection.csv", flush=True)

def validate_built_probability():
    merged_df = compare_to_test_data()
    
    cm = make_total_confusion_matrix(merged_df)
    plot_confusion_matrices_for_tiles(merged_df)
    calculate_evaluation_metrics(merged_df, cm)
    
      
write_pixels_from_nedbygging_tiffs(tiff_paths)
validate_built_probability()

