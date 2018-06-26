Read me 

1) Download Part2 folder.
2) Open command prompt and navigate till the Part2/CosineSimilarity folder.
3) Run the following command.


spark-submit --class "MainApplication" ml_2.11-0.1.jar article1.csv TopiModelling2.csv topiclabelling_result_cosinesim.csv

Note: 
The  input file containes a very small dataset to make sure the algorithm doesnt take long time to execute for evaluation purpose.