
import os
import math
import json
import requests
import numpy as np
import geopandas
from shapely.geometry import Point
from PIL import Image

from fastapi import FastAPI, HTTPException
from typing import TypedDict, Literal

from abm_result_to_gdf import convert_abm_result_to_gdf

import geo_to_raster

from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
citypyo_url = os.getenv("CITYPYO_URL")




class ScenarioProperties(TypedDict):
    bridge_hafencity: bool
    underpass_veddel_north: bool
    main_street_orientation: Literal["horizontal", "vertical"]
    blocks: Literal["open", "closed"]
    roof_amenities: Literal["random", "complementary"]

class AbmResultRequest(TypedDict):
    userid: str
    scenario_properties: ScenarioProperties
    result_format: Literal["png", "web-frontend"]


class PngResponse(TypedDict):
    bbox_sw_corner: list
    img_width: int
    img_height: int
    bbox_coordinates: list
    image_base64_string: str



def get_poi_geojson(scen_props: ScenarioProperties):
    filename = scen_props["main_street_orientation"] + "_" + scen_props["roof_amenities"]
    filename = filename + ".geojson"

    cwd = os.getcwd()

    with open(cwd + "/points_of_interest/" + filename) as fp:
        geojson = json.load(fp)
        fp.close()

    return geojson


# returns the project area as gdf
def get_project_area_from_citypyo(userid)-> geopandas.GeoDataFrame:
    response = requests.get(
        citypyo_url + "getLayer", 
        json= {
            "userid": userid,
            "layer": "project_area"
        }
    )

    if not response.status_code == 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"CityPyo Error when fetching project area: {response.text}"
        )

    
    project_area_json = response.json()

    gdf = geopandas.GeoDataFrame.from_features(
        project_area_json["features"],
        crs=project_area_json["crs"]["properties"]["name"]
    )

    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def get_result_from_citypyo(
    userid: str,
    scenario_properties: ScenarioProperties
):
    query = "abmScenario"
    data = {
        "userid": userid,
        "scenario_properties": scenario_properties
    }

    response = requests.post(
        citypyo_url + "getLayer/" + query,
        json=data
    )

    if not response.status_code == 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"CityPyo Error: {response.text}"
        )

    response = response.json()

    return response["simulationResult"]



async def rasterize(project_area, gdf, resolution, hour):
    """
    filters the result gdf for result in the current hour
    and rasterizes them to a grid with "resolution"
    and boundaries of project_area
    The grid is returned as numpy array
    """
    start_time = hour * 3600  # map to seconds
    end_time = (hour+1) * 3600  # map to seconds

    # create a new gdf with a subset of the data for the given hour
    gdf_subset = gdf[gdf["timestamp"].between(
        start_time, end_time+1, inclusive="left")]

    raster = geo_to_raster.rasterize_gdf(
        gdf_subset,
        "agent_count",
        resolution,
        "add",
        project_area.total_bounds,
        project_area.crs
    )

    # mirror along x-axis (wind and noise provide results like that and it works for unity..)
    raster = np.flipud(raster)

    return raster


async def raster_to_img(raster, resolution):
    """
    converts the raster into a image. 
    Rescales image by resolving each pixel in to an array of resolution*resolution
    in oder to scae the image
    """
    im = Image.fromarray(raster)
    im = im.resize((raster.shape[0] * resolution , raster.shape[1] * resolution), Image.Resampling.LANCZOS)
    
    return im
    

async def normalize_result(hourly_result: np.ndarray, resolution: int):
    """
    normalizes results to #people / (m² * h)
    then multiplies them by 10 rounds to ints in order to create PNG compatible values.
    """
    def normalize(x):
        return x / (resolution*resolution)
    
    def to_int(x):
        return int(round(x* 10))

    normalize_np_array = np.vectorize(normalize)
    to_int_np_array = np.vectorize(to_int)

    hourly_result = normalize_np_array(hourly_result)
    hourly_result = to_int_np_array(hourly_result)
    hourly_result = hourly_result.astype(np.uint8)

    return hourly_result



@app.post("/abm_result_as_pngs")
async def abm_result_as_pngs(
    userid: str,
    scenario_properties: ScenarioProperties,
    resolution: int = 5
) -> list[PngResponse]:

    """
    Obtains resultset and project area from citypyo
    Splits result set into subsets for each hour of the simulated time
    Returns array of png (base64)
    """
    
    project_area = get_project_area_from_citypyo(userid)
    abm_result = get_result_from_citypyo(userid, scenario_properties)
    gdf = convert_abm_result_to_gdf(abm_result)
    gdf = geopandas.clip(gdf, project_area) # clip to project area
    simulation_duration_in_h = math.floor(gdf["timestamp"].max() / 3600)

    rasterized_results = [
        await rasterize(project_area, gdf, resolution, hour) for hour in range(simulation_duration_in_h)
    ]
    rasterized_results = [
        await normalize_result(raster, resolution) for raster in rasterized_results
    ]
    result_images = [
        await raster_to_img(raster, resolution) for raster in rasterized_results
    ]
   
    
    # prepare response object
    geo_bounds = geo_to_raster.get_bounds_coords_list(project_area.total_bounds),
    sw_corner = geo_to_raster.get_south_west_corner_coords_gdf(project_area.total_bounds),

    response_obj = []
    for hour, img in enumerate(result_images):
        img_width, img_height = img.size
        response_obj.append({
            "image_base64_string": geo_to_raster.img_to_base64_png(img),
            "hour": hour + 8
        })


    return {
        "points_of_interest": get_poi_geojson(scenario_properties),
        "results": response_obj,
        "coordinates": {
            "bbox_sw_corner": sw_corner[0],
            "bbox_coordinates": geo_bounds[0],
            "img_width": img_width,
            "img_height": img_height
        }
    }
