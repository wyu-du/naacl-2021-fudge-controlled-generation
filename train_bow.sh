
export device_id=1

CUDA_VISIBLE_DEVICES=$device_id python -u main.py \
--task topic \
--data_dir convai2_data \
--save_dir ckpt/convai2/future_word_retrain_predictor \
--num_workers 1 \
--batch_size 16 \
--epoch_max_len 100000 \
--validation_freq 10  \
--lr 2e-4 \
--epochs 20 \
--glove_file convai2_data/glove.840B.300d.txt > future_word_retrain_predictor.log

