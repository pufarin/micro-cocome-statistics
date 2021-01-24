# micro-cocome-statistics

## Possible ML model:
* Communication between services (CBS) : RESTful, Database, Messaging
* API-Gateway (APG) : True , False
* Communication between the client and the application (CBCA) : Syncronus, Asyncronus
* Communication between the API-Gateway and the other services (CBAPGS) : RESTful, Messaging

## Implementation Versions

| Version Nr | Name                              |
|------------|-----------------------------------|
| v01        | master_1_to_1_db                  |
| v02        | master_one_db                     |
| v03        | api_gateway_1_to_1_db             |
| v04        | api_gateway_one_db                |
| v05        | pub_sub_1_to_1_db                 |
| v06        | pub_sub_one_db                    |
| v07        | message_bus_1_to_1_db             |
| v09        | orchestrate_api_gateway_1_to_1_db |
| v10        | orchestrate_pub_sub_1_to_1_db     |

## Model Description
| Version | CBS_REST | CBS_DB | CBS_MESSAGING | APG | CBCA_SYNC | CBCA_ASYNC | CBAPGS_REST | CBAPGS_MESSAGING | ELAPSED_TIME |
|---------|----------|--------|---------------|-----|-----------|------------|-------------|------------------|--------------|
| v01     | 1        | 0      | 0             | 0   | 1         | 0          | 0           | 0                |              |
| v02     | 0        | 1      | 0             | 0   | 1         | 0          | 0           | 0                |              |
| v03     | 1        | 0      | 0             | 1   | 1         | 0          | 1           | 0                |              |
| v04     | 0        | 1      | 0             | 1   | 1         | 0          | 1           | 0                |              |
| v05     | 1        | 0      | 0             | 1   | 0         | 1          | 0           | 1                |              |
| v06     | 0        | 1      | 0             | 1   | 0         | 1          | 0           | 1                |              |
| v07     | 0        | 0      | 1             | 1   | 0         | 1          | 0           | 1                |              |
| v09     | 1        | 0      | 0             | 1   | 1         | 0          | 1           | 0                |              |
| v10     | 0        | 0      | 1             | 1   | 0         | 1          | 0           | 1                |              |

## How to run the code
* Have Python 3 running on your machine
* Download the github repo onto your machine
* Change the value of the __absolutPath__ to the path of the measurements dir on your machine
* you should be able to run the scripts