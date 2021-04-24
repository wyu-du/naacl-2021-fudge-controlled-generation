
export device_id=4

CUDA_VISIBLE_DEVICES=$device_id python -u evaluate_topic.py \
--ckpt ckpt/convai2/future_word_retrain_predictor/model_best.pth.tar \
--dataset_info ckpt/convai2/future_word_retrain_predictor/dataset_info \
--condition_file convai2_data/convai2_gen_dev.txt \
--condition_lambda 4.0 \
--verbose \
--precondition_topk 200 \
--topk 10 \
--sample_size 3 \
--max_sample_batch 1 \
--length_cutoff 80 \
--log_file convai2_preds.log


