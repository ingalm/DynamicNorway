import ee
import geopandas as gpd

ee.Authenticate()
ee.Initialize(project='ee-ingvildalmasbakk')

# 

year = 2017
startMonth = 6
endMonth = 8
CLEAR_THRESHOLD = 0.6
S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B11','B12']
QA_BAND = 'cs_cdf'

# Read aoi from .shp file names tiles_test.shp
aoi_shapefile = 'tiles_test.shp'
aoi_gdf = gpd.read_file(aoi_shapefile)

aoi_geometries = aoi_gdf.geometry.apply(lambda geom: ee.Geometry(geom.__geo_interface__)).tolist()
aoi_fc = ee.FeatureCollection(aoi_geometries)

# Load Norway boundary
fylker = ee.FeatureCollection('users/zandersamuel/NINA/Vector/Norway_administrative_fylker_2022')
norway_geom = fylker.geometry()

# Define grid size
grid_size = 0.5  # Grid cells of 0.5° by 0.5°

# Function  to create a grid of rectangular cells over a given region
def create_grid(region, grid_size):
    # Get bounds of the region
    bounds = region.bounds()
    
    # Get the minimum and maximum latitudes and longitudes of the region
    bounds_info = bounds.getInfo()
    
    # Extract the coordinates of the bounding box
    coordinates = bounds_info['coordinates'][0]
    min_lon = coordinates[0][0]
    min_lat = coordinates[0][1]
    max_lon = coordinates[2][0]
    max_lat = coordinates[2][1]

    # Create lists of longitude and latitude values for the grid
    lons = [min_lon + i * grid_size for i in range(int((max_lon - min_lon) / grid_size) + 1)]
    lats = [min_lat + i * grid_size for i in range(int((max_lat - min_lat) / grid_size) + 1)]
    print(f"Grid size: {len(lons)}x{len(lats)}")
    
    grid_cells = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            cell = ee.Geometry.Rectangle(
                [lons[i], lats[j], lons[i+1], lats[j+1]]
            )
            grid_cells.append(cell)
    
    return ee.List(grid_cells)

# Create the grid over the Norway geometry
grid_cells = create_grid(norway_geom, grid_size)

# Function to apply cloud mask based on Cloud Score +
def apply_cloud_mask(img):
    cloud_mask = img.select(QA_BAND).gte(CLEAR_THRESHOLD)
    masked_image = img.updateMask(cloud_mask)
    return masked_image

print(f"Processing month: {startMonth} to {endMonth}")
print("Fetching sentinel-2 images...")
s2_summer = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filter(ee.Filter.calendarRange(startMonth, endMonth, 'month'))
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .filterBounds(norway_geom)
            .select(S2_BANDS))

print("Fetching cloud score plus...")
cs_col = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        .filter(ee.Filter.calendarRange(startMonth, endMonth, 'month'))
        .filter(ee.Filter.calendarRange(year, year, 'year'))
        .filterBounds(norway_geom))

print("Linking collections...")
linked_collection = s2_summer.linkCollection(cs_col, [QA_BAND])

    
total_tiles = 0 

print(f"Number of images in linked collection: {linked_collection.size().getInfo()}")

print("Processing grid cells...")
for i, grid_cell in enumerate(grid_cells.getInfo()):
    if i not in [83, 169, 335, 355, 806, 1050]:
        continue
    try:
        print(f"\n🔍 Processing grid cell {i + 1}...")
    
        # Check if the grid cell intersects with Norway's boundary
        grid_geom = ee.Geometry(grid_cell)
        if not grid_geom.intersects(norway_geom).getInfo():
            print(f"Grid cell {i + 1} is outside Norway's boundary. Skipping.")
            continue
        
        # Clip the linked collection to the current grid cell
        linked_tile = linked_collection.filterBounds(grid_geom)

        s2_count = linked_tile.size().getInfo()
        
        if s2_count == 0:
            print(f"No S2 images for grid cell {i + 1}")
            continue
        print(f" {s2_count} images found for grid cell {i + 1}")
        
        # Apply cloud mask to each image in the linked collection
        cloud_masked_images = linked_tile.map(apply_cloud_mask)

        sorted_images = cloud_masked_images.select(S2_BANDS) # Select only the bands of interest
        
        # Create the median composite from the cloud-masked images
        median = sorted_images.median()
        
        s2_masked = median.clip(aoi_fc)

        # Export
        task = ee.batch.Export.image.toDrive(
            image=s2_masked,
            description=f's2_{year}_grid_cell_{i + 1}',
            folder=f's2_{year}',
            fileNamePrefix=f's2_{year}_grid_cell_{i + 1}',
            scale=10,
            region=grid_geom,
            crs='EPSG:32632',
            maxPixels=1e13
        )
        
        task.start()
        total_tiles += 1
        print(f"Export started for grid cell {i + 1}")

    except Exception as e:
        print(f"Error with grid cell {i + 1}: {e}")
        continue


print(f"Total tiles processed: {total_tiles}")