# vector data
import geopandas
import pandas as pd
from shapely.geometry import Polygon, Point

# rasterization
import numpy as np
from numpy.typing import NDArray


# returns a list of point features
def convert_abm_result_to_gdf(abm_result) -> geopandas.GeoDataFrame:
    data = {
        "lat": [], "lon": [], "timestamp": []
    }

    for agent_data in abm_result:
        for path_id, coords in enumerate(agent_data["path"]):
            data["lon"].append(float(coords[0]))
            data["lat"].append(float(coords[1]))
            data["timestamp"].append(agent_data["timestamps"][path_id])

    # create a normal pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    # "agent_count" for each datapoint is 1. 
    # "agent_count" will be aggregated in rasterization
    df["agent_count"] = 1  

    # Create a geodataframe by parsing the lat/lon information into a geometry col.
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326"

    )

    return gdf
