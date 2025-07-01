import ee

# Initialize your GEE module
ee.Initialize(project='placeholder-project-id')

# Load Norway counties for spatial filtering
fylker = ee.FeatureCollection('users/zandersamuel/NINA/Vector/Norway_administrative_fylker_2022')

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
    .reproject(proj.atScale(10)))

# Apply remapping to grunnkart image
remappedGrunnkart = grunnkart.remap(fromValues, toValues)

# Load Dynamic World training data
dw = ee.ImageCollection('projects/wri-datalab/dynamic_world/v1/DW_LABELS')
dwNorway = dw.filterBounds(fylker)

# Function to build export stack
def getImgStack(img):
    s2id = img.get('S2_GEE_ID').getInfo()

    s2img = ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{s2id}').select(S2_BANDS)
    dwimg = ee.Image(f'GOOGLE/DYNAMICWORLD/V1/{s2id}').select('label')
    
    cloudMask = dwimg.gte(0)

    s2img = s2img.updateMask(cloudMask)

    maskedDwLabel = img.rename('dwLabel').updateMask(cloudMask);
    maskedGrunnkart = remappedGrunnkart.rename('grunnkartLabel').updateMask(cloudMask);

    exportImg = (s2img
        .addBands(maskedDwLabel)
        .addBands(maskedGrunnkart)
        .uint16())
    
    return exportImg

# Generate export tasks
total_images = dwNorway.size().getInfo()
start_index = 0
imgList = dwNorway.toList(1000)

for i in range(start_index, total_images):
    try: 
        imgSel = ee.Image(imgList.get(i))

        # Check for valid S2_GEE_ID
        s2id_val = imgSel.get('S2_GEE_ID').getInfo()
        if s2id_val is None or s2id_val == '':
            print(f"Skipping tile {i} — missing S2_GEE_ID.")
            continue

        expImg = getImgStack(imgSel).set('dw_id', imgSel.get('dw_id'))
        
        task = ee.batch.Export.image.toDrive(
            image=expImg,
            description=f'dw_stack_{i}',
            folder='Master/dw_norway_new',
            scale=10,
            region=imgSel.geometry()
        )
        task.start()
        print(f"Export started for tile {i}")
    except Exception as e:
        print(f"Error exporting tile {i}: {e}")
        continue
