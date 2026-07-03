sudo pip3 uninstall flash-attn timm

pip3 install -e ".[train]"
pip3 install flash-attn --no-build-isolation
pip3 install ipdb tensorboard openai openpyxl datasets pytesseract decord mamba-ssm causal-conv1d
pip3 install -U pillow

pip3 install matplotlib yake
pip3 install open_clip_torch diffusers[torch] ezcolorlog
pip3 install timm==0.9.16
pip3 install opencv-python
pip3 install datasets pandas
