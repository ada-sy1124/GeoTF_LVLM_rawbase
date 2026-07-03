# GeoTF LVLM

This folder keeps `LVLM_Interpretation_raw` as the base project and adds a
separate GeoTF path for geolocation attribution.

Raw files and model folders are copied whole. GeoTF-specific files are:

- `main_geo_tf.py`
- `args_geo_tf.py`
- `geo_tf_geo.py`
- `geo_tf_score.py`
- `methods_geo_tf.py`
- `geo_tf_eval.py`
- `model_adapters/`

Main loop:

```text
original image -> greedy natural coordinate answer y*
extract first valid latitude, longitude span
build WGS84-aware coordinate token weights
mask -> masked image -> weighted teacher-forced GeoTF loss
maximize GeoTF loss with iGOS-style mask regularization
render deletion heatmap and deletion evaluation
```

Run step 0 once for each tokenizer/model:

```bash
python3 00_check_tokenizer_coordinate_format.py
```

Edit `MODEL_PATH` at the top of `00_check_tokenizer_coordinate_format.py`
before switching models. This step only records token ids, token strings, and
decoded pieces for coordinate strings; it does not feed an artifact into
`main_geo_tf.py`.

Run GeoTF attribution:

```bash
python3 main_geo_tf.py \
  --model llava \
  --model_path liuhaotian/llava-v1.5-13b \
  --data_path /path/to/data.jsonl \
  --image_folder /path/to/images \
  --output_dir /path/to/output \
  --size 32 \
  --L1 1.0 \
  --L2 0.1 \
  --L3 10.0 \
  --gamma 1.0 \
  --ig_iter 10 \
  --iterations 5 \
  --momentum 5 \
  --lr 10
```

Input JSONL:

```json
{"id": "sample_0001", "image_path": "sample.jpg"}
```

`image_path` can be absolute or relative to `--image_folder`.
