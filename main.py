import os

os.system("python src/data_ingestion.py")
print('Data Ingestion done')
os.system('python src/data_preprocessing.py')
print('Data Preprocessing Done')
os.system('python src/data_hyper_parameter_tuning.py')
print('Data Hyperparameter Tuning Done')
os.system('python src/Model_evaluation.py')
print('Data Model Evaluation Done')