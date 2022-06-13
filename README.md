# functionalscope-abm-result-provider

Fast api app to    
- fetch api abm results from cityPyo
- convert them to geopandas geodataframe
- rasterize that geodataframe
- return a set of pngs containing a heatmap per simulated hour

## Install
"> docker build -t abm-result-provider ." 

## Env
create .env file with CITYPYO_URL param

## Run
"> $ docker run -p 80:80 --env-file=./.env abm-result-provider"