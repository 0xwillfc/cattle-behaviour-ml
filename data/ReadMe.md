
# Japanese Black Beef Cow Behavior Classification Dataset
This dataset contains tri-axial accelerometer sensor data with thirteen different labeled cow behaviors. This data was gathered with a 16bit +/- 2g Kionix KX122-1037 accelerometer attached to the neck of six different Japanese Black Beef Cows (`cow1.csv`-`cow6.csv`) at a cow farm of Shinshu University in Nagano, Japan on the 12th of June, 2020. 
	
The data gathering took place over the course of one day in which the cows were allowed to roam freely in two different areas, namely, a grass field and farm pens, while being filmed with Sony FDR-X3000 4K video cameras.

The timestamps of the video and accelerometer data were matched while human observers which included behavior experts and non-experts labeled the data from the video footage. The labeling and data gathering took a total of 69 person-hours.

567 minutes of unlabeled data were parsed into 197 minutes of high-quality labeled data comprising thirteen behaviors by means of majority voting with three annotators. The time per behavior in number of samples (@25Hz) and their respective descriptions are shown in the following table:


|     | Cow 1 | Cow 2 | Cow 3 | Cow 4 | Cow 5 | Cow 6 | Sum    | Description                     |
|:----|:---|:------|:------|:------|:------|:------|:-------|:--------------------------------|
| RES | 35814 | 47059 | 20501 | 15735 | 11025 | 19996 | 150130 | Resting in standing position    |
| RUS | 1620  | 25930 | 11156 | 14523 | 0     | 0     | 53229  | Ruminating in standing position |
| MOV | 6376  | 8437  | 7532  | 17248 | 4846  | 5760  | 50199  | Moving                          |
| GRZ | 2416  | 2199  | 0     | 2707  | 2442  | 7849  | 17613  | Grazing                         |
| SLT | 204   | 0     | 10654 | 0     | 0     | 0     | 10858  | Salt licking                    |
| FES | 6809  | 0     | 0     | 0     | 1125  | 0     | 7934   | Feeding in stancheon            |
| DRN | 1176  | 0     | 1300  | 0     | 0     | 0     | 2476   | Drinking                        |
| LCK | 0     | 0     | 649   | 297   | 0     | 356   | 1302   | Licking                         |
| REL | 0     | 360   | 0     | 404   | 0     | 0     | 764    | Resting in lying position       |
| URI | 239   | 0     | 383   | 0     | 0     | 0     | 621    | Urinating                       |
| ATT | 57    | 50    | 0     | 62    | 0     | 197   | 366    | Attacking                       |
| ESC | 0     | 0     | 0     | 128   | 0     | 0     | 128    | Escaping                        |
| BMN | 0     | 54    | 0     | 0     | 0     | 0     | 54     | Being mounted                   |
| ETC | 105917     | 103084    | 129297     | 62064    | 53922     | 100571     | 554855     | Other behaviors                  |
| BLN | 151249     | 82599    | 88431     | 111744    | 61544     | 45128     | 540695     | Data without video, no label                  |
| Sum | 311876 | 269772 | 269903 | 224912 | 134904 | 179857 | 1391224 |                               


Accelerometer sampling rate was set to 25Hz.
The data is split into six .csv files which represents each of the 6 cows above. The columns of these files are defined as follows:


| TimeStamp_UNIX [-] |TimeStamp_JST [-]| AccX [g]   | AccY [g]   | AccZ [g]    | Label [-]  |
|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| GPS Timestamp in UNIX | GPS Timestamp in JST| X-axis acceleration | Y-axis acceleration | Z-axis acceleration | labeled behavior |

The gathering of this data with these cows was reviewed and approved by the Institutional Animal Care and Use Committee of Shinshu University.

## Data logger open source software 
Software developed for the data logger that was used to gather this dataset, Sony's IoT development board SPRESENSE, CXD5602PWBMAIN1. The function of this data logger is to write inertia sensor data along with timestamps. Timestamp data is corrected with GPS signal. Available in Arduino development environment.

https://github.com/cattleuser/Spresense_EVK-701_RECORDER

## Publications using this dataset
[1] Li, Chao, et al. "Data Augmentation for Inertial Sensor Data in CNNs for Cattle Behavior Classification." IEEE Sensors Letters 5.11 (2021): 1-4.

[2] Bartels, Jim, et al. "A 216 microW, 87% Accurate Cow Behavior Classifying Decision Tree on FPGA With Interpolated Arctan2." 2021 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 2021.