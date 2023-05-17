# STAD
Implementation of article: **Space-temporal Attention Decomposition (STAD): An Interpretable Head Space-temporal Decomposition  Module for Multivariate Time Series**.

**Abstract**: As forecasting multivariate time series (MTS) becomes increasingly important in various disciplines, industry, and academia have employed traditional statistical and deep learning-based methods to predict MTS. However, traditional methods struggle to manage large-scale, heterogeneous data effectively, whereas deep learning-based techniques cannot interpretively capture spatio-temporal information between series. We propose an interpretable head decomposition mechanism, Space-temporal Attention Decomposition (STAD), for MTS forecasting. STAD mechanism consists of spatial and temporal structures with global and local components and high-frequency and low-frequency components, which reflect the potential mechanism of time series. STAD estimates the weights and trajectories of each component using an attention mechanism, making it a flexible and data-driven method that can enhance the accuracy of MTS models. In addition, STAD can be connected to most MTS networks to improve their performance interpretively, and experiments on three public datasets demonstrate its efficacy in enhancing MTS forecasting.

-----
## Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0

------
## Data


- **BikeNYC**: The dataset consists of time series data that represents the overall demand for shared bicycles in specific regions of New York City. The map is grid into 16$\times$

- **Electricity**: The original dataset includes electricity consumption data for 370 points/customers. We removed 34 outliers containing extreme values. In addition, we calculated the hourly average electricity consumption for each point and treated it as the time series to be modeled.

- **PeMSD7**: The PeMSD7 dataset is collected from sensor stations deployed in the highway system of major urban areas in California, which are used to monitor traffic speeds. The dataset contains speed data from over 1,000 sensors, sampled every 30 seconds. To make the dataset more manageable and easier to work with, we aggregated the data into 30-minute intervals using average pooling. 

- **Traffic**: This repository contains hourly data spanning 48 months (2015-2016) obtained from the California Department of Transportation. The data pertains to the degree of road usage (ranging from 0 to 1) as measured by various sensors installed on the freeways of the San Francisco Bay area. As the dataset is too large, we take the first 100 locations from the original dataset for training. 

- **Solar**: The dataset provided in this study consists of the solar power production records from 137 photovoltaic (PV) plants located in the state of Alabama during the year 2006. The dataset is comprised of measurements taken every 10 minutes, providing a detailed picture of solar power production patterns throughout the day. As the dataset is too large, we take the first 70 locations from the original dataset for training. 
  The following datasets are sourced entirely from the ERA5 dataset \cite{hersbach2018era5}. This dataset compiles hourly climate data from diverse locations worldwide since 1940. We have chosen four variables, namely SP (Surface Pressure), SKT (Skin Temperature), STL1 (Soil Temperature Level 1), and MSL (Mean Sea Level Pressure), and have randomly selected 100 locations for each indicator. We have gathered data at 0:00, 6:00, 12:00, and 18:00 every day from 2017 to 2022, covering a total of six years.
- **SP (Surface pressure)**: This variable refers to the pressure exerted by the atmosphere at the Earth's surface, including over land, sea, and inland water. It is a measure of the total weight of air above a specific point on the Earth's surface. This parameter is measured in Pascals (Pa).

- **SKT (Skin temperature)**: This variable represents the Earth's surface temperature and is known as the skin temperature. It is a theoretical temperature that is needed to balance the Earth's energy budget. The calculation of skin temperature is not uniform over land and sea. The unit of this parameter is kelvin (K).

- **STL1(Soil temperature level 1)**: This variable refers to the temperature of the first layer of soil, specifically at its midpoint. The first layer shall be at 7cm and soil temperature is set at the midpoint of each layer. Soil temperature is measured worldwide, including over the ocean, but grid points with a land-sea mask value less than 0.5 are excluded to account for water-covered regions. The unit for this parameter is kelvin (K).

- **MSL(Mean sea level pressure)**: This variable refers to the atmospheric pressure at the Earth's surface, which is standardized to the average sea level. The standard unit of measurement for this parameter is pascals (Pa).
\end{itemize}

**Link**: https://pan.baidu.com/s/16PsgGe_BWaBgNhDpxEExfQ?pwd=stad 
**Keyword**: stad

-----
## Arguments

- `model`: backbone architecture (default=0, 0 for wavenet /1 for tcn /2 for transformer).  
- `n_his`: number of input steps.
- `hidden_channels`: number of hidden channels.  
- `n_his`: number of input steps.  
- `n_layers`: number of hidden layers.
- `st1`: whether to use ST(A)D.
- `st2`: whether to use ST-Norm.
- `STNorm_n`: number of STD.
- `TSNorm_n`: number of TSD.
- `n_train`: number of training batch size.
- `n_val`: number of validating batch size.
- `n_test`: number of testing batch size.
- `n`: number of locations.
- `attention`: whether to add attention mechanism to the model.
- `n_slots`: number of slots.
- `filename`: dataset name.
- `real_data`: whether the data is real data or simulated data.

-----
#### Model training and evaluation for WaveNet with `STD`
```
python main.py --st1 1 --st2 0 --attention 0
```

#### Model training and evaluation for WaveNet with `STAD`
```
python main.py --st1 1 --st2 0 --attention 1
```
