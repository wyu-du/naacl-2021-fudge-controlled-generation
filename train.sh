
export device_id=1

CUDA_VISIBLE_DEVICES=$device_id python -u main.py \
--task intent \
--data_dir dailydialog_data \
--save_dir ckpt/intent/intent_retrain_predictor \
--num_workers 1 \
--batch_size 16 \
--epoch_max_len 1000000 \
--validation_freq 1 \
--lr 2e-5 \
--epochs 10 > intent_retrain_predictor.log

