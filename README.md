# tradeNetwork ReadMe
Joint work with Anthony Bonato, Vincent Luong and Kyne Santos

# Directory Structure

## data
Contains trade network data files.
- `Oil Trade k3 graph.csv` - edge list with year label of top 3 import partners and top 3 export partners for each country
- `TradeData_1988_1999.csv` - US Comtrade data on oil exports and imports from 1988 to 1999
- `TradeData_2000_2011.csv` - US Comtrade data on oil exports and imports from 2000 to 2011
- `TradeData_2012_2023.csv` - US Comtrade data on oil exports and imports from 2012 to 2023
- `TradeData_2024_2025.csv` - US Comtrade data on oil exports and imports from 2024 to 2025
- `Oil Trade full graph.csv` - edge list with year and weight label. Source is exporting country
- `Oil Trade stat table.csv` - lists top 5 countries by pagerank and weighted indegree from 1988 to 2025, along with the number of nodes, edges, clustering coefficient, number of connected components, and average path length

## `oil_k3_graph_models.ipynb`
Notebook file containing code used to run the model prediction on the k3 oil network.

This notebook utilizes the python files `four_profile.py` and `three_profile.py` for the implementation of getting the 
4-profile of a graph.

## `oil trade centrality analysis.ipynb`
Notebook file containing code to compute centrality measures from 1988 to 2025. Contains time series plots of the top 20 oil exporting countries by pagerank, indegree, and the spearmann correlations of how these rankings change by year.
