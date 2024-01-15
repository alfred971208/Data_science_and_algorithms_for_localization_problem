This GitHub project is based on my thesis on location optimization of agricultural facilities using algorithms and data analysis. It focuses on determining the optimal location for a Federal Inspection Type (TIF) slaughterhouse in Hidalgo, Mexico, maximizing the production and export of beef cattle.
Requires Python with specific libraries detailed in the project files. Ideal to be executed in Google Colab, making it easy to use and access.
This project is a project capable of localizing any type of industry through science behind. 
The idea of this project is to generate through data science techniques, data analysis and algorithms, a plant location, for this I use data from INEGI and SAGARPA that later I try to predict a production by means of a ML model, so that with the algorithms I can locate an optimal location.
It is important to know that non-Euclidean distances are being used to treat distances, however, the use of an API such as Google Maps is essential for the location to have feasibility in real life, it should be clarified that the VRP needs that the company that uses this needs to have a vision of occupied vehicles, as well as their capacities, likewise the data shown by the different algorithms contemplate simulated customers, it is best to have a list of customers or potential customers proposed by the company, as well as what if the database is extensive occupy instead of pandas, for example; apache, mysql or whatever the company decides. 
For this example we will use data for the location of a federal inspection type slaughterhouse (TIF) in Hidalgo, considering that the logic behind it is to build by means of treated data a ML model that predicts the production of heads of cattle, so that later given the logic that a federal inspection type slaughterhouse experts the meat, it is necessary to know the export production for it in "analisis_datos.py, analisis_topologico_y_multifractal.py, modelo_predictivo.py, datos_analisis_pronostico.py, eficacia_del_modelo.py and lógica_exportacion_importacion.py " will show the whole process done considering that the average consumption will be subtracted minus the average production predicted giving us certain data that will help us to subsequently buy them with customer data that for this example were occupied municipalities of the highest HDI in Hidalgo.
With this in mind, we will use models of comparison between clusters (Kmeans) and center of gravity to build a model for the location of the slaughterhouse, where the logic will be first to use distribution and collection centers for these cattle in order to reduce costs and not make several slaughterhouses and then compare this data with the best municipalities to locate the slaughterhouse, this with government articles that mention previous aid for the export of these cattle in certain municipalities.  
Now the idea of the location is to use a modified TSP simulating 1x1 merchandise, a normal TSP, a modified VRP with clusters since we do not have data of how many trucks, of what size and what capacity we have, as well as a NSGA-II genetic algorithm for the first location of the centers and then the best location for the slaughterhouse. Now the file names are mentioned in Spanish under the names already mentioned with .py extension in this repository to replicate those results.
The data occupied to run the codes are in the Files folder.

Instructions for use:
Run the codes in Google Colab installing the necessary dependencies first and place well where the files will be placed in your working module, as well as follow the idea before proposed to visualize the results.

Visualizations:


![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/9cc03c0d-d466-4c6a-8c65-399f2ff062bc)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/46558dc0-e32c-4faa-9545-bff2b4768d62)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/4751d4ed-5f24-4742-b2ed-b9b510d80503)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/fddc3fc5-d7e9-45a5-8cfd-366be298ca5e)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/33de3c85-fa92-4821-8541-deddf4684a46)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/1feca258-cdbd-4108-b6b4-1d589ad57ee9)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/460f5b89-2df9-46a5-b1a0-f96aebe429ca)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/3c0a1eae-a4a8-4f5d-b7f6-116db2f61d74)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/5b184bdb-a33d-492d-babd-a1375fd67fcc)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/3095b068-f59d-49f2-b783-8c1cb7413f1a)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/d4469637-25ca-4e18-9c78-84280aa1ea10)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/78b05044-1052-442e-8d71-5d1fc0e3f245)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/ac97895b-2c18-45d2-ab70-a4e66a9f92ba)

![image](https://github.com/alfred971208/Data_science_and_algorithms_for_localization_problem/assets/29002971/fa71ace7-e7c4-4d43-be38-f25041853337)


License:
This project is under the GNU General Public License v3.0, available in the License.txt file, ensuring the open use, modification and distribution of the code.

Contact:
For contact and more information, you can refer to lagz.consultant@gmail.com with Ing. Luis Alfredo González Zamora.

FAQ or Common Questions:

How can I adapt the project to another location?
Review the input parameters and adjust the geographic and demographic data according to your area of interest.

What data is needed to replicate this analysis?
You will need demographic and geographic data similar to those used in this project, such as those provided by INEGI and SAGARPA.

Can I use different data sets?
Yes, the project is adaptable to different datasets as long as they maintain a similar structure and content.

Is it necessary to have advanced knowledge in Python?
A basic understanding of Python is helpful, but the code is designed to be accessible to users with different skill levels.

How can I contribute to the project?
You can contribute by improving the code, proposing new features or fixing bugs.

Is the project suitable for educational purposes?
Absolutely, it is an excellent resource for learning about location optimization and data analysis.
