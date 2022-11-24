# Gravity_Model

This repository shall be used to determine whether or not the raised incidence model by Diggle & Rowlingson (1994) can be used for tracing back foodborne disease outbreaks. Originally, the model is used to investigate the elevation in risk for a specified disease in relation to possible environmental factors. Here, instead of a transmission via air, the disease is transmitted through the shopping behavior of people.  
Since there are no such real outbreaks available to test the method (because their either not traced back to a certain chain or not available on the needed level of granularity), we generate artificial outbreaks to test the method. Firstly, to show relevancy we implement the method on real locational data of supermarkets and real population data in Hamburg in 2011. Secondly, to determine under which conditions the model works best, we use stylized outbreaks that are based on stylized geographical rooms and stylized locations of retail chain stores.

For both, a gravity model is used to determine the flows between the retail chains and the consumers. Based on these typical outbreaks can be generated. To calibrate the gravity model the technique by Hyman (1969) is used, while the flows are determined using the Furness method (1965).

## Example Case Hamburg in 2011

- read_census_Data_Hamburg.ipynb: This file reads the census data of 2011 that are in grid cells of 100m x 100m. The data is taken from the statistical office Nord (https://www.statistik-nord.de/fileadmin/maps/zensus2011_hh/index.html)

- read:shops.ipynb: This file reads the retail shops and their revenues in 2011. Data taken from Sarah (TradeDimensions?)

- gravity_model.ipynb: using population data and retail store location data, this file tries to implement the gravity model. Inspiration can be taken from Marcel Fuhrman's supply chain model helper which implements the gravity model as well. (https://gitlab.com/DjMaFu/supplychainmodulator)

## Stylized Outbreaks

The stylized outbreaks shall be created based on stylized geographical rooms and stylized retail store locations. These shall be based on different levels of entropy.

## References

Diggle, Peter J., and Barry S. Rowlingson. “A Conditional Approach to Point Process Modelling of Elevated Risk.” Journal of the Royal Statistical Society. Series A (Statistics in Society) 157, no. 3 (1994): 433. https://doi.org/10.2307/2983529.

Hyman, G. M. “The Calibration of Trip Distribution Models.” Environment and Planning A: Economy and Space 1, no. 1 (June 1969): 105–12. https://doi.org/10.1068/a010105.

Furness, K. P. (1965). Time Function Iteration. Traffic Engineering and Control 7.7, 458–460.
