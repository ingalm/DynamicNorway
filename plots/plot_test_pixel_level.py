import os
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
import rasterio
from shapely import Polygon

df = pd.read_csv("data/test_pixel_level.csv") 

# Load a map of Norway for WorldAdminitrative boundaries
norway = gpd.read_file("data/WorldAdministrativeBorders/world-administrative-boundaries.shp")
norway = norway.geometry

norway = norway.to_crs("EPSG:4326")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

# Plotting
fig, ax = plt.subplots(figsize=(10, 12))
norway.plot(ax=ax, color='white', edgecolor='black')  # Base map

# Plot stable (green) and loss (red)
gdf[gdf.label == "stable"].plot(ax=ax, markersize=5, color='green', label="Stable")
gdf[gdf.label == "loss"].plot(ax=ax, markersize=5, color='red', label="Loss")


def get_aoi_from_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        aoi_projection = src.crs
        
        print(f"AOI Projection: {aoi_projection}")
        print(f"Type of aoi_projection: {type(aoi_projection)}")
        
        # Ensure aoi_projection is a valid CRS object
        if isinstance(aoi_projection, str):
            aoi_projection = CRS.from_user_input(aoi_projection)
        
        # Create a Polygon for the AOI 
        aoi_polygon = Polygon([
            (bounds[0], bounds[1]), 
            (bounds[2], bounds[1]), 
            (bounds[2], bounds[3]), 
            (bounds[0], bounds[3])  
        ])
    
    return aoi_polygon, aoi_projection

# Overlay rectangles from the AOI tiles
aoi_dir = "data/aoi" # Directory containing AOI .tif files
for filename in os.listdir(aoi_dir):
    if filename.endswith(".tif"):
        # Get AOI polygon and projection for each .tif file
        aoi_polygon, aoi_projection = get_aoi_from_tiff(os.path.join(aoi_dir, filename))
        
        # Transform the polygon to WGS84 (EPSG:4326)
        aoi_gdf = gpd.GeoDataFrame([aoi_polygon], columns=['geometry'], crs=aoi_projection)
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
        
        # Plot the AOI polygon
        aoi_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linestyle='-', linewidth=2)


proxy_aoi = Line2D([0], [0], marker='o', color='blue', label='AOI', markerfacecolor='blue', markersize=5, linestyle=None)
proxy_stable = Line2D([0], [0], marker='o', color='green', label='Stable', markerfacecolor='green', markersize=5, linestyle=None)
proxy_loss = Line2D([0], [0], marker='o', color='red', label='Loss', markerfacecolor='red', markersize=5, linestyle=None)


plt.title("Land Cover Change in Norway")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(handles=[proxy_aoi, proxy_stable, proxy_loss])
plt.savefig("plots/norway_pixel_map_aoi.png", dpi=300, bbox_inches='tight')
plt.show()
