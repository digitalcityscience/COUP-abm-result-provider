import base64

# vector data
import geopandas
from shapely.geometry import Polygon, Point, box

# rasterization
import numpy as np
from numpy.typing import NDArray

from geocube.rasterize import rasterize_image
from odc.geo.geobox import GeoBox
from rasterio.enums import MergeAlg

from PIL import Image

# to raster
from io import BytesIO

from typing import Literal


MergeAlgorithm = Literal["add", "replace"]

# returns an np_array raster with png img like data [0-255]
def rasterize_gdf(
    gdf:geopandas.GeoDataFrame,
    property_to_burn:str,
    resolution: int,
    merge_algorithm: MergeAlgorithm,
    custom_bounds_coords: list = [],
    custom_bounds_crs: str = "EPSG:4326",
    ) -> NDArray:

    # set CRS to metric (Germany)
    crs = "EPSG:25832"
    gdf = gdf.to_crs(crs)
    
    # set merge algorithm
    if merge_algorithm == "replace":
        merge_alg = MergeAlg.replace
    elif merge_algorithm == "add":
        merge_alg = MergeAlg.add
    else:
        raise ValueError(
            f"merge algorithm can be add or replace. Got: {merge_algorithm}"
    )

    # create custom bounding box if coordinates provided.
    if list(custom_bounds_coords):
        bounding_box = box(*custom_bounds_coords)
        gpd_box = geopandas.GeoSeries(bounding_box)
        gpd_box = gpd_box.set_crs(custom_bounds_crs)
        gpd_box = gpd_box.to_crs(crs)
        bounds_coords = gpd_box.total_bounds
    else:
    # otherwise use bounding box of data
        bounds_coords = gdf.total_bounds

    geobox = GeoBox.from_bbox(
        bbox=bounds_coords,
        crs=crs,
        resolution=resolution
    )

    raster_data = rasterize_image(
        geometry_array = gdf.geometry,  # A geometry array of points.
        data_values = gdf[property_to_burn].to_numpy(),
        geobox=geobox,
        fill=0,
        merge_alg=merge_alg
    )

    # create a np array with dtype uint8 from rasterized data
    img = raster_data.astype(np.uint8)
    # for debugging: 
    # print("unique values in image data ", np.unique(img))

    return img


# gets [x,y] of the south west corner of the bbox.
# might only work for european quadrant of the world
def get_south_west_corner_coords_gdf(gdf_bounds) -> list:
    left, bottom, _, _ = gdf_bounds
    
    sw_point = Point([left, bottom])

    return list(sw_point.coords)


# returns a list of coordinates for the bounding box polygon
def get_bounds_coords_list(bounds) -> list:
    left, bottom, right, top = bounds

    pol = Polygon([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
        [left, top]
        ]
    )

    return list(pol.exterior.coords)

def img_to_base64_png(im: Image) -> str:
    # create a pillow image, save it and convert to base64 string
    output_buffer = BytesIO()
    im.save(output_buffer, "PNG")
    
    byte_data = output_buffer.getvalue()
    base64_bytes = base64.b64encode(byte_data)
    base64_string = base64_bytes.decode('utf-8')

    return base64_string
