EXPERIMENT_MODEL="EXP_TSMixer"
TRAIN_DATA_PATH="DATA_EEG/EEG_PREP_ROBUST_TRAIN"             
TEST_DATA_PATH="DATA_EEG/TEST_50"               
VAL_DATA_PATH="DATA_EEG/VAL_70"           

EXPERIMENT_DIR="EXPERIMENTS/EXP_0411_JLB"
PREDICTION_LENGTH=96 
CONTEXT_LENGTH=336 

BATCH_SIZE_TRAIN=32  
BATCH_SIZE_VAL=32
BATCH_SIZE_TEST=1

EARLY_STOPPING=15
VAL_CHECK_INTERVAL=6000
LIMIT_VAL_BATCHES=1 # run all batches for validation 
MAX_EPOCHS=5
LIMIT_TEST_BATCHES=64
LIMIT_TRAIN_BATCHES=100000
MAX_SAMPLES_VAL=9600

NUM_WORKERS=8


echo "RUNNING EXPERIMENT 0133"
python Exp_0133_run.py \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_model $EXPERIMENT_MODEL \
    --model_name "TSMixer" \
    --prediction_length $PREDICTION_LENGTH \
    --context_length 336 \
    --train_data_path $TRAIN_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_val $BATCH_SIZE_VAL \
    --batch_size_test $BATCH_SIZE_TEST \
    --early_stopping $EARLY_STOPPING \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --max_samples_val $MAX_SAMPLES_VAL \
    --num_workers $NUM_WORKERS \

echo "finished TSMixer 336"

python Exp_0133_run.py \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_model $EXPERIMENT_MODEL \
    --model_name "TSMixer" \
    --prediction_length $PREDICTION_LENGTH \
    --context_length 512 \
    --train_data_path $TRAIN_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_val $BATCH_SIZE_VAL \
    --batch_size_test $BATCH_SIZE_TEST \
    --early_stopping $EARLY_STOPPING \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --max_samples_val $MAX_SAMPLES_VAL \
    --num_workers $NUM_WORKERS \

echo "finished TSMixer 512"

python Exp_0133_run.py \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_model $EXPERIMENT_MODEL \
    --model_name "TSMixer" \
    --prediction_length $PREDICTION_LENGTH \
    --context_length 720 \
    --train_data_path $TRAIN_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_val $BATCH_SIZE_VAL \
    --batch_size_test $BATCH_SIZE_TEST \
    --early_stopping $EARLY_STOPPING \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --max_samples_val $MAX_SAMPLES_VAL \
    --num_workers $NUM_WORKERS \

echo "finished TSMixer 720"

python Exp_0133_run.py \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_model $EXPERIMENT_MODEL \
    --model_name "TSMixer" \
    --prediction_length $PREDICTION_LENGTH \
    --context_length 1024 \
    --train_data_path $TRAIN_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --batch_size_train $BATCH_SIZE_TRAIN \
    --batch_size_val $BATCH_SIZE_VAL \
    --batch_size_test $BATCH_SIZE_TEST \
    --early_stopping $EARLY_STOPPING \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --max_samples_val $MAX_SAMPLES_VAL \
    --num_workers $NUM_WORKERS \


echo "finished TSMixer 1024"