
export device_id=4

CUDA_VISIBLE_DEVICES=$device_id python -u evaluate_topic.py \
--model_string ~/baseline-dialogue/src/microsoft/DialoGPT-medium_convai2_raw_base_2021-04-26-10-21-06/checkpoint-47839 \
--ckpt ckpt/convai2/future_word_retrain_predictor/model_best.pth.tar \
--dataset_info ckpt/convai2/future_word_retrain_predictor/dataset_info \
--condition_file convai2_data/convai2_gen_dev.txt \
--condition_lambda 4.0 \
--precondition_topk 200 \
--topk 10 \
--sample_size 1 \
--max_sample_batch 1 \
--length_cutoff 80 \
--log_file convai2_ft_preds.log > convai2_ft_preds.log


