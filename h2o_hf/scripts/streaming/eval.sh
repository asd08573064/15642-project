method=$1
if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=0 python run_streaming.py \
        --inference_type heavy_hitter \
        --heavy_hitter_size 48 \
        --recent_size 2000
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=0 python run_streaming.py \
     --inference_type streaming
else
    CUDA_VISIBLE_DEVICES=0 python run_streaming.py \
    --inference_type lsh
fi
