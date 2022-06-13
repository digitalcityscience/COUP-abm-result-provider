""" 
def make_seaborn_heatmap():
    import seaborn as sns
    import geoplot as gplt
    import matplotlib.pyplot as plt

    #ax = gplt.polyplot(project_area, projection=gplt.crs.AlbersEqualArea(), zorder=1)
    # todo set extent
    gplt.kdeplot(
        gdf_subset["geometry"],
        cbar=True, cmap='Reds',
        shade=True,
        clip=project_area,
        vmax=60000
    ) 
    #, vmin=0, vmax=10) #, ax=ax)

 """