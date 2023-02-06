#! /bin/bash

#python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/validation/run2_exp1.txt
#python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/validation/run2_exp2.txt
#python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/validation/run2_exp3.txt
#python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/validation/run2_exp4.txt
#python main.py -c main_config_exp5.ini 2>&1 | tee ./logs/validation/run2_exp5.txt

python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/study/run2_exp1.txt
python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/study/run2_exp2.txt
python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/study/run2_exp3.txt
python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/study/run2_exp4.txt
python main.py -c main_config_exp5.ini 2>&1 | tee ./logs/study/run2_exp5.txt

#python main.py -c main_config_exp1.ini 2>&1 | tee ./logs/testset_study/exp1.txt
#python main.py -c main_config_exp2.ini 2>&1 | tee ./logs/testset_study/exp2.txt
#python main.py -c main_config_exp3.ini 2>&1 | tee ./logs/testset_study/exp3.txt
#python main.py -c main_config_exp4.ini 2>&1 | tee ./logs/testset_study/exp4.txt
#python main.py -c main_config_exp5.ini 2>&1 | tee ./logs/testset_study/exp5.tx