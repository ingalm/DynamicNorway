import ee

# Initialize your Earth Engine module
ee.Initialize(project='placeholder-project-id')

CLEAR_THRESHOLD = 0.6
QA_BAND = 'cs_cdf'

# Load Norway counties for spatial filtering
fylker = ee.FeatureCollection('users/zandersamuel/NINA/Vector/Norway_administrative_fylker_2022')
proj_utm = ee.Projection('EPSG:32633').atScale(10)


S2_BANDS = ['B2','B3','B4', 'B5', 'B6', 'B7','B8','B11','B12'] # Sentinel-2 bands used


# Value lookup for Dynamic World labels
dwLabelDict = {
    'water': 1,
    'trees': 2,
    'grass': 3,
    'flooded_vegetation': 4,
    'crops': 5,
    'shrub_and_scrub': 6,
    'built': 7,
    'bare': 8,
    'snow_and_ice': 9
}

# Value lookup for grunnkart labels
grunnkartLabelDict = {
    "Bebyggelse/samferdsel": 1,
    "Dyrket mark": 2,
    "Grasmark": 3,
    "Skog": 4,
    "Hei og åpen vegetasjonb": 5,
    "Lite vegetert mark": 6,
    "Våtmark": 7,
    "Elver/bekker": 8,
    "Innsjøer/tjern": 9,
    "Marine bukter og brakkvann": 10,
    "Svaberg, kyststrender og dyner": 11,
    "Åpent hav": 12,
    "Uklassifisert": 99,
}

# Define remapping from grunnkartLabelDict to dwLabelDict
remapDict = {
    1: 7,  # built
    2: 5,  # crops
    3: 3,  # grass
    4: 2,  # trees
    5: 6,  # shrub_and_scrub
    6: 8,  # bare
    7: 4,  # flooded_vegetation
    8: 1,  # water
    9: 1,  # water
    10: 1, # water
    11: 8, # bare
    12: 1, # water
    99: 99  # unclassified
}

# Function to apply cloud mask based on Cloud Score +
def apply_cloud_mask(img):
    cloud_mask = img.select(QA_BAND).gte(CLEAR_THRESHOLD)
    masked_image = img.updateMask(cloud_mask)
    return masked_image

# Convert mapping into lists for remap function
fromValues = list(remapDict.keys())
toValues = list(remapDict.values())

# Import "grunnkart for arealregnskap" image collection
grunnkartCol = ee.ImageCollection('users/zandersamuel/NINA/Raster/Norway_grunnkart_okosys')
proj = grunnkartCol.first().projection()
grunnkart = grunnkartCol.mosaic().select(0)  # First band is level 1

# Reduce resolution to 10m using mode reducer
grunnkart = (grunnkart
    .reproject(proj.atScale(3))
    .reduceResolution(ee.Reducer.mode())
    .reproject(proj.atScale(10))
    .reproject(proj_utm))

# Apply remapping to grunnkart image
remappedGrunnkart = grunnkart.remap(fromValues, toValues).rename('grunnkart')

# Load Dynamic World training data
# Used for spatial geometry filtering
dw = ee.ImageCollection('projects/wri-datalab/dynamic_world/v1/DW_LABELS')
dwNorway = dw.filterBounds(fylker)

dw_size = dwNorway.size().getInfo()
imgList = dwNorway.toList(1000)

# Date range
start_date = '2024-06-01'
end_date = '2024-08-31'

for i in range(0, dw_size):
    try:
        dw_img = ee.Image(imgList.get(i))
        geometry = dw_img.geometry()
        s2_id = dw_img.get('S2_GEE_ID').getInfo()

        if not s2_id:
            print(f"Skipping tile {i} — missing S2_GEE_ID.")
            continue

        tile_id = s2_id.split('_')[-1].replace('T', '')

        print("s2_id:", s2_id)
        print("tile_id:", tile_id)
        
        print(f"Fetching image collections")

        # Sentinel-2 summer composite
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start_date, end_date)
              .filterBounds(geometry)
              .select(S2_BANDS))
   
        # Cloud Score + collection
        cs = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
              .filterDate(start_date, end_date)
              .filterBounds(geometry))
        
        linked_collection = s2.linkCollection(cs, [QA_BAND])
        
        print(f"Adding cloud mask to collection for tile {i}")
        
        cloud_masked_collection = linked_collection.map(apply_cloud_mask)
        
        print(f"Tile {i} — S2 image collection size: {cloud_masked_collection.size().getInfo()}")
        if cloud_masked_collection.size().getInfo() == 0:
            print(f"Tile {i} — No images found for tile {tile_id}.")
            continue


        composite = cloud_masked_collection.median().reproject(proj_utm).clip(dw_img.geometry())
        grunnkart_reproj = remappedGrunnkart.reproject(proj_utm)
        
        band_names = composite.bandNames().getInfo()
        composite = composite.select(S2_BANDS)

        # Clip to the training tile
        stacked = composite.addBands(grunnkart_reproj).clip(geometry).uint16()

        if geometry.area(1).getInfo() < 1000:
            print(f"Tile {i} — Empty or tiny DW tile. Skipping.")
            continue

        band_names = stacked.bandNames().getInfo()
        print(f"Tile {i} — Band names: {band_names} (Count: {len(band_names)})")

        # Export
        task = ee.batch.Export.image.toDrive(
            image=stacked,
            description=f'summer_stack_tile_{i}',
            folder=f'FINAL_NIBIO_summer_2024',
            fileNamePrefix=f'summer_s2_tile_{i}',
            region=geometry,
            scale=10,
            crs='EPSG:32633', 
            maxPixels=1e13
        )
        task.start()
        print(f"Started export for tile {i}: {tile_id}")

    except Exception as e:
        print(f"Error at tile {i}: {e}")
        continue