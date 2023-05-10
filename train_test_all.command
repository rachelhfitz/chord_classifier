python3 train.py --observation_folder data/training_set/basic_triads --model_folder models/basic_triads
python3 train.py --observation_folder data/training_set/complex_piano --model_folder models/complex_piano
python3 train.py --observation_folder data/training_set --model_folder models/testset_all

python3 test.py --testset_folder data/test_set/basic_triads --model_folder models/basic_triads
python3 test.py --testset_folder data/test_set/basic_triads --model_folder models/complex_piano
python3 test.py --testset_folder data/test_set/basic_triads --model_folder models/testset_all
python3 test.py --testset_folder data/test_set/complex_piano --model_folder models/basic_triads
python3 test.py --testset_folder data/test_set/complex_piano --model_folder models/complex_piano
python3 test.py --testset_folder data/test_set/complex_piano --model_folder models/testset_all